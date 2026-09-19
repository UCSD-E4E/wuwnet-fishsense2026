"""Finding the calibration target in an image, and saying which brick is which.

The running bond rules out a checkerboard detector, so this is the replacement.
It is built around what the target actually shows rather than around what a
board would: the dark seam between bricks swallows the black ones, so a wall
reads as isolated light quadrilaterals on a dark field, and finding those is
easier than fitting seam lines, not harder.

Correspondence is the real problem, and it is solved per wall with a homography
rather than per point. A wall is planar, so one detected marker brick matched to
one model marker brick already determines a homography from that wall's own
coordinates into the image; projecting the rest of the wall through it and
counting agreements either confirms the guess or kills it. That is cheap enough
to try exhaustively -- a few hundred hypotheses -- which matters, because a local
patch of the wall cannot identify itself (see the model module's tests) and so
nothing more local than a whole-wall registration would work.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration_model import (
    MARKER_COLOURS,
    MODEL_IO,
    Face,
    exposed_faces,
    visible_corners,
)

#: Hue centres, in OpenCV's 0-179 scale, for the saturated LDraw colours.
HUE_DEGREES = {1: 107.0, 4: 3.0, 2: 70.0, 14: 27.0}  # blue, red, green, yellow


@dataclass(frozen=True)
class Quad:
    """One detected brick face: its colour, and its outline in the image."""

    colour: int
    corners: np.ndarray  # (4, 2) image pixels, ordered around the face

    @property
    def centre(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def classify_colours(
    image,
    value_floor: int = 55,
    saturation_floor: int = 125,
    white_value: int = 110,
) -> np.ndarray:
    """Label every pixel with an LDraw colour, or -1 for seam and background.

    Deliberately a blunt instrument. The thresholds are exposed because they are
    the one part of this module that will not survive the move from renders to
    photographs unchanged -- white balance, strobe falloff and the plastic's
    specular highlight all land here, and nowhere else.

    Black is *not* labelled. A black brick against a dark seam has no boundary
    to find, so treating black as background is not a limitation being accepted
    but a description of what the image contains.
    """
    # Classified unblurred, deliberately. The seam between two bricks in
    # adjacent courses is the chamfer alone -- half a millimetre, a few pixels
    # at best and sub-pixel obliquely -- and any smoothing here closes it, which
    # merges a whole staggered column of same-coloured bricks into one blob.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0].astype(float), hsv[..., 1], hsv[..., 2]
    labels = np.full(image.shape[:2], -1, np.int16)

    pale = (saturation < saturation_floor) & (value >= white_value)
    labels[pale] = 15

    saturated = (saturation >= saturation_floor) & (value >= value_floor)
    distance = np.stack(
        [np.abs((hue - centre + 90) % 180 - 90) for centre in HUE_DEGREES.values()]
    )
    nearest = np.argmin(distance, axis=0)
    codes = np.array(list(HUE_DEGREES))
    close = np.take_along_axis(distance, nearest[None], axis=0)[0] < 18.0
    labels[saturated & close] = codes[nearest][saturated & close]
    return labels


def detect_quads(
    image,
    min_area_px: float = 150.0,
    approx_tolerance: float = 0.03,
    fill_floor: float = 0.85,
) -> List[Quad]:
    """Find brick faces as four-sided blobs of one colour.

    Two properties of the target dictate how this is done, and both were found
    by running it rather than by reasoning about it.

    Blobs are labelled **4-connected**. In a running bond a brick and the brick
    a course above it meet at a single corner, so same-colour faces touch
    diagonally; under the 8-connectivity that `findContours` implies, a whole
    staggered column of white bricks merges into one zigzag blob. The seam is a
    real gap, so 4-connectivity is not a trick -- it is the correct reading of
    the image at the resolution where the gap is a pixel wide.

    Blobs with six vertices are **split**, because a brick on the target's
    vertical edge shows two faces at once and projects to a chevron rather than
    a quad. Dropping those would be worse than it sounds: the marker bricks that
    make correspondence possible sit at the wall ends, which is precisely where
    the target's edges are.

    What a face is *not* tested for is rectangularity -- see `_fill`.
    """
    labels = classify_colours(image)
    quads: List[Quad] = []
    for colour in np.unique(labels):
        if colour < 0:
            continue
        mask = (labels == colour).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, blobs = cv2.connectedComponents(mask, connectivity=4)
        for index in range(1, count):
            blob = (blobs == index).astype(np.uint8)
            area = float(blob.sum())
            if area < min_area_px:
                continue
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, approx_tolerance * perimeter, True)
            polygon = approximation.reshape(-1, 2).astype(np.float64)
            for candidate in _split_polygon(polygon):
                if abs(cv2.contourArea(candidate.astype(np.float32))) < min_area_px:
                    continue
                if _fill(blob, candidate) < fill_floor:
                    continue
                quads.append(Quad(int(colour), _order_corners(candidate)))
    return quads


def _fill(blob: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of the fitted quad that the blob actually occupies.

    This is the test that a detected face is one brick rather than several, and
    it has to be fill rather than shape: perspective shears a brick face into a
    parallelogram, so any measure of rectangularity rejects exactly the oblique
    views a calibration set exists to provide. A merged staggered column, by
    contrast, fills about half of the quad drawn around it whatever its shear.
    """
    stencil = np.zeros_like(blob)
    cv2.fillConvexPoly(stencil, np.round(quad).astype(np.int32), 1)
    covered = float(stencil.sum())
    return float((stencil & blob).sum() / covered) if covered > 0 else 0.0


def _split_polygon(polygon: np.ndarray) -> List[np.ndarray]:
    """A quad as itself; a six-sided chevron as the two quads it folds from."""
    if len(polygon) == 4:
        return [polygon]
    if len(polygon) != 6:
        return []
    best, halves = np.inf, []
    for start in range(3):
        first = polygon[[start, start + 1, start + 2, start + 3]]
        second = polygon[[(start + 3) % 6, (start + 4) % 6, (start + 5) % 6, start]]
        if not all(cv2.isContourConvex(h.astype(np.float32)) for h in (first, second)):
            continue
        cut = float(np.linalg.norm(polygon[start] - polygon[(start + 3) % 6]))
        if cut < best:
            best, halves = cut, [first, second]
    return halves


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Counter-clockwise from the corner nearest the top left, in image axes."""
    centre = corners.mean(axis=0)
    angle = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    ordered = corners[np.argsort(angle)]
    start = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -start, axis=0)


def _wall_frame(faces: Sequence[Face]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An orthonormal ``(origin, along, up)`` for the plane a wall lies in."""
    normal = faces[0].normal
    up = np.array([0.0, 0.0, 1.0])
    along = np.cross(up, normal)
    origin = np.mean([f.centre for f in faces], axis=0)
    return origin, along, up


def _wall_groups(path: Path = MODEL_IO):
    groups = {}
    for face in exposed_faces(path, sides_only=True):
        groups.setdefault(tuple(np.round(face.normal, 6)), []).append(face)
    return groups


@dataclass
class Correspondence:
    """Matched object and image points for one view, ready for `calibrateCamera`."""

    object_points: np.ndarray  # (N, 3) model millimetres
    image_points: np.ndarray  # (N, 2) pixels
    faces: Tuple[Face, ...]  # the model faces they came from, four points each

    def __len__(self) -> int:
        return len(self.object_points)


def _wall_coordinates(faces: Sequence[Face]):
    """Each face's visible corners in the wall's own 2D frame, plus the frame."""
    origin, along, up = _wall_frame(faces)
    corners = {}
    for face in faces:
        local = visible_corners(face) - origin
        corners[id(face)] = np.stack([local @ along, local @ up], axis=-1)
    return corners, (origin, along, up)


def match(
    quads: Sequence[Quad],
    camera_intrinsics,
    path: Path = MODEL_IO,
    centre_tolerance: float = 0.45,
    min_faces: int = 4,
    agreement_deg: float = 6.0,
) -> Correspondence:
    """Decide which model face each detected quad is, wall by wall.

    A wall is planar, so a single detected marker brick paired with a single
    model marker brick of the same colour already fixes a homography from that
    wall's coordinates into the image -- four corners are four correspondences.
    Projecting the rest of the wall through it and counting agreements then
    either confirms the pairing or kills it, at a few hundred hypotheses per
    wall, which is cheap.

    Seeding from *markers* is what keeps that count small: a marker colour
    appears eight or ten times on the whole target where white appears fifty.
    Seeding from a local neighbourhood would not work at all, because the
    interior of a wall is a periodic two-colour bond and a patch of it cannot
    say where it is.

    Every wall is then made to prove itself. The hypothesis search allows a
    reflection, because the detector's corner order and the model's differ by a
    handedness that depends on the view; but a reflection also lets a wall
    *facing away* from the camera land on the same quads, which scores well and
    is nonsense. So each wall's pairing is put through planar PnP, and kept only
    if the resulting pose leaves that wall facing the camera and agrees with the
    best-scoring wall. Without this the recovered pose comes back mirrored --
    around 175 degrees out -- with a perfectly healthy-looking match count.
    """
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=float)
    detected = list(quads)
    candidates = []

    for normal_key, wall in _wall_groups(path).items():
        local, _ = _wall_coordinates(wall)
        markers = [f for f in wall if f.colour in MARKER_COLOURS]

        # Both handednesses are carried all the way to the pose test, and the
        # score is not allowed to choose between them. The detector orders
        # corners counter-clockwise in image axes, where y points down, while
        # the model orders them counter-clockwise in the wall's own frame; the
        # two differ by a reflection. A reflected match reprojects *well* -- it
        # places the camera at its mirror image through the wall plane, under a
        # pixel of residual -- so it reliably outscores the true one. Only the
        # outward normal can tell them apart.
        for flip in (1, -1):
            best_score, best_pairs = 0, []
            for face in markers:
                source = local[id(face)].astype(np.float32)
                for quad in detected:
                    if quad.colour != face.colour:
                        continue
                    for roll in range(4):
                        target = np.roll(quad.corners[::flip], roll, axis=0).astype(np.float32)
                        homography = cv2.getPerspectiveTransform(source, target)
                        if not np.isfinite(homography).all():
                            continue
                        pairs = _score(homography, wall, local, detected, centre_tolerance)
                        if len(pairs) > best_score:
                            best_score, best_pairs = len(pairs), pairs
            if best_score < min_faces:
                continue

            # Grow the match. The seed homography comes from one brick's four
            # corners extrapolated across a whole wall, so it is good enough to
            # find near neighbours and little else; refitting on everything it
            # found and rescoring pulls in the rest.
            for _ in range(5):
                source = np.vstack([local[id(f)] for f, _ in best_pairs]).astype(np.float32)
                target = np.vstack([c for _, c in best_pairs]).astype(np.float32)
                homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
                if homography is None:
                    break
                grown = _score(homography, wall, local, detected, centre_tolerance)
                if len(grown) <= len(best_pairs):
                    break
                best_pairs = grown

            pose = _wall_pose(best_pairs, np.array(normal_key), camera_intrinsics)
            if pose is not None:
                candidates.append((len(best_pairs), pose, best_pairs))

    if not candidates:
        return Correspondence(np.empty((0, 3)), np.empty((0, 2)), ())

    candidates.sort(key=lambda item: -item[0])
    reference = candidates[0][1]
    object_points, image_points, matched_faces = [], [], []
    for _, pose, pairs in candidates:
        if _pose_gap(pose, reference) > agreement_deg:
            continue
        for face, corners in pairs:
            object_points.append(visible_corners(face))
            image_points.append(corners)
            matched_faces.append(face)
    return _reject_outliers(
        Correspondence(np.vstack(object_points), np.vstack(image_points), tuple(matched_faces)),
        camera_intrinsics,
    )


def _reject_outliers(
    correspondence: Correspondence, camera_intrinsics, tolerance_px: float = 3.0
) -> Correspondence:
    """Drop whole faces that a single pose cannot explain.

    Per-wall scoring is local: it is satisfied by a face landing near where the
    wall's own homography says it should, which a face one course out can also
    do. A pose fitted across every wall at once does not have that freedom, so
    it is what catches them.

    Faces are dropped whole. A face whose four corners disagree with the pose is
    not three good corners and one bad one; it is the wrong brick.
    """
    if len(correspondence) < 8:
        return correspondence
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        correspondence.object_points,
        correspondence.image_points,
        np.asarray(camera_intrinsics, dtype=float),
        None,
        reprojectionError=tolerance_px,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return correspondence
    projected, _ = cv2.projectPoints(
        correspondence.object_points, rvec, tvec, np.asarray(camera_intrinsics, dtype=float), None
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - correspondence.image_points, axis=1)
    keep = residual.reshape(-1, 4).max(axis=1) <= tolerance_px
    if keep.sum() < 2:
        return correspondence
    mask = np.repeat(keep, 4)
    return Correspondence(
        correspondence.object_points[mask],
        correspondence.image_points[mask],
        tuple(f for f, k in zip(correspondence.faces, keep) for _ in range(4) if k),
    )


def _wall_pose(pairs, normal, camera_intrinsics):
    """Planar PnP for one wall, kept only if the wall ends up facing the camera.

    A plane admits two poses that reproject equally well, and a reflected match
    admits a third that is simply wrong; the outward normal settles all of it.
    """
    object_points = np.vstack([visible_corners(f) for f, _ in pairs])
    image_points = np.vstack([c for _, c in pairs])
    ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_points, image_points, camera_intrinsics, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    best = None
    for rvec, tvec, error in zip(rvecs, tvecs, np.ravel(errors)):
        rotation, _ = cv2.Rodrigues(rvec)
        centre_model = -rotation.T @ tvec.reshape(3)
        facing = np.dot(normal, object_points.mean(axis=0) - centre_model) < 0
        if facing and (best is None or error < best[2]):
            best = (rotation, tvec.reshape(3), float(error))
    return best


def _pose_gap(pose, reference) -> float:
    """Angle between two recovered rotations, in degrees."""
    relative = pose[0] @ reference[0].T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def _score(homography, wall, local, detected, centre_tolerance):
    """Face-to-quad pairings implied by a homography, with corners in order."""
    pairs = []
    for face in wall:
        projected = cv2.perspectiveTransform(
            local[id(face)].reshape(1, 4, 2).astype(np.float32), homography
        ).reshape(4, 2)
        scale = np.linalg.norm(projected - projected.mean(axis=0), axis=1).mean()
        centre = projected.mean(axis=0)
        for quad in detected:
            if quad.colour != face.colour:
                continue
            if np.linalg.norm(quad.centre - centre) > centre_tolerance * scale:
                continue
            # Corner order is set by the projection, not by the detector's own
            # arbitrary starting corner.
            cost = np.linalg.norm(projected[:, None, :] - quad.corners[None, :, :], axis=2)
            order = [int(np.argmin(cost[i])) for i in range(4)]
            if len(set(order)) != 4:
                continue
            pairs.append((face, quad.corners[order]))
            break
    return pairs
