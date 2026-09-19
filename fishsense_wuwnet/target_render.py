"""Synthetic views of the LEGO calibration target.

The detector has to exist before the target has been photographed, and it has to
be scoreable afterwards, which needs ground truth a photograph cannot supply:
which lattice point is which, and where each one landed to the sub-pixel. So the
target is rendered from the same `.io` file the detector is tested against, and
the render returns the projected lattice alongside the image.

This is deliberately a *geometry* simulator and not an attempt at photorealism.
It gets the projection right, including the flat port, because that is what the
detector's accuracy is measured against; it gets the appearance only as right as
it needs to be for the detector's front end to have something to bite on -- flat
colour, a dark seam where bricks meet, optional blur and noise. A detector tuned
on these images will still need its colour front end re-tuned on real ones, and
the tests say so rather than pretending otherwise.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .calibration_model import (
    BRICK_CLEARANCE_MM,
    CHAMFER_MM,
    MODEL_IO,
    LatticePoint,
    exposed_faces,
    lattice_points,
)
from .refraction import FlatPort, alpha_azimuth_to_pixel, project_water_points

#: LDraw's published colour values, as BGR for OpenCV.
LDRAW_BGR = {
    0: (29, 19, 5),  # black
    15: (255, 255, 255),  # white
    1: (191, 85, 0),  # blue
    4: (9, 26, 201),  # red
    2: (65, 120, 35),  # green
    14: (55, 205, 242),  # yellow
}

#: The dark line where two bricks meet, reproduced by insetting each face over a
#: dark background rather than by drawing lines on top of it -- which is what
#: physically happens, and it matters that the two insets differ. Side to side
#: the inset is half the moulding clearance; top to bottom the bricks touch and
#: the line is the chamfer alone. A detector that finds the joint *centre* is
#: unbiased; one that finds a brick's visible edge is short by these amounts,
#: and rendering them separately is what lets the tests tell those apart.
SEAM_BGR = (18, 16, 14)
INSET_ACROSS_MM = BRICK_CLEARANCE_MM / 2.0 + CHAMFER_MM
INSET_UP_MM = CHAMFER_MM

#: Studs are drawn on upward faces only. They are not features -- they localise
#: poorly, which is why the lattice is built from brick corners -- but they are
#: drawn so that a detector run on these images has to reject them, as it will
#: have to on a photograph.
STUD_RADIUS_MM = 2.4

_QUADRANT = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


@dataclass
class TargetView:
    """A rendered view, with the ground truth a detector is scored against."""

    image: np.ndarray  # (H, W, 3) BGR
    lattice: Tuple[LatticePoint, ...]
    pixels: np.ndarray  # (N, 2) projected lattice points
    visible: np.ndarray  # (N,) on a camera-facing wall and inside the image

    @property
    def seen(self) -> np.ndarray:
        """The visible lattice points, as ``(M, 2)`` pixels."""
        return self.pixels[self.visible]

    @property
    def seen_model(self) -> np.ndarray:
        """Their model coordinates, as ``(M, 3)`` millimetres -- the object points."""
        return np.array([p.position for p in self.lattice])[self.visible]


def _projector(camera_intrinsics, rotation, translation, port: Optional[FlatPort]):
    """A model-millimetres to pixels map, through air or through a flat pane.

    The port case is not a distortion applied after a pinhole projection: behind
    a flat pane there is no single centre of projection, so the field angle has
    to be solved per point. `project_water_points` does that, which is why this
    takes points rather than returning a matrix.
    """

    def project(points_mm):
        points = np.asarray(points_mm, dtype=float).reshape(-1, 3)
        camera = points @ rotation.T + translation  # millimetres, camera frame
        if port is None:
            depth = camera[:, 2:3]
            pixels = camera[:, :2] / depth
            pixels = pixels * [camera_intrinsics[0, 0], camera_intrinsics[1, 1]]
            pixels = pixels + [camera_intrinsics[0, 2], camera_intrinsics[1, 2]]
        else:
            alpha, azimuth = project_water_points(camera / 1000.0, port)
            pixels = alpha_azimuth_to_pixel(alpha, azimuth, camera_intrinsics)
        return pixels, camera[:, 2]

    return project


def render(
    camera_intrinsics,
    rotation,
    translation,
    size=(4000, 3000),
    port: Optional[FlatPort] = None,
    path: Path = MODEL_IO,
    background=(70, 60, 48),
    blur_px: float = 1.2,
    noise: float = 2.0,
    studs: bool = True,
    seed: Optional[int] = 0,
) -> TargetView:
    """Render the target.

    `rotation` and `translation` take model millimetres into the camera frame,
    and `translation` is in millimetres too. `port` renders the view as it would
    be seen through a flat pane; `None` renders it in air, which is the case the
    target is actually used in -- the port matters here because the detector is
    also what a refractive-reconstruction check would run on.
    """
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(3)
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=float)
    width, height = size
    project = _projector(camera_intrinsics, rotation, translation, port)

    # The camera centre in model coordinates, for back-face culling. Valid for
    # the port case too: refraction bends rays but does not let a camera see a
    # surface that faces away from it.
    centre_model = -rotation.T @ translation

    faces = exposed_faces(path)
    order = []
    for face in faces:
        if np.dot(face.normal, face.centre - centre_model) >= 0.0:
            continue
        _, depth = project(face.centre[None])
        order.append((float(depth[0]), face))
    order.sort(key=lambda item: -item[0])  # painter's algorithm, far to near

    image = np.full((height, width, 3), background, np.uint8)
    for _, face in order:
        outer, depth = project(face.corners)
        if np.any(depth <= 0.0):
            continue
        # Inset by the same millimetre on both axes. Scaling the corner offsets
        # by one factor would inset the short axis in proportion, making the
        # horizontal seams three times thinner than the vertical ones -- which
        # would hand the detector an asymmetry the real target does not have.
        du = (face.corners[1] - face.corners[0]) / 2.0
        dv = (face.corners[3] - face.corners[0]) / 2.0
        inset_u = INSET_UP_MM if abs(du[2]) > abs(du[0]) + abs(du[1]) else INSET_ACROSS_MM
        inset_v = INSET_UP_MM if abs(dv[2]) > abs(dv[0]) + abs(dv[1]) else INSET_ACROSS_MM
        du = du * (1.0 - inset_u / np.linalg.norm(du))
        dv = dv * (1.0 - inset_v / np.linalg.norm(dv))
        inset = face.centre + _QUADRANT[:, 0:1] * du + _QUADRANT[:, 1:2] * dv
        inner, _ = project(inset)
        shade = 0.55 + 0.45 * abs(float(np.dot(face.normal, rotation[2])))
        colour = tuple(int(c * shade) for c in LDRAW_BGR[face.colour])
        cv2.fillConvexPoly(image, np.round(outer).astype(np.int32), SEAM_BGR, cv2.LINE_AA)
        cv2.fillConvexPoly(image, np.round(inner).astype(np.int32), colour, cv2.LINE_AA)
        if studs and float(face.normal[2]) > 0.5:
            for su in (-1.5, -0.5, 0.5, 1.5):
                for sv in (-0.5, 0.5):
                    ring = face.centre + su * du / 2.0 + sv * dv
                    edge = ring + STUD_RADIUS_MM * du / np.linalg.norm(du)
                    (c0, e0), _ = project(np.stack([ring, edge])), None
                    radius = int(round(np.linalg.norm(c0[1] - c0[0])))
                    if radius >= 1:
                        cv2.circle(image, tuple(np.round(c0[0]).astype(int)), radius,
                                   tuple(int(c * shade * 0.93) for c in LDRAW_BGR[face.colour]),
                                   -1, cv2.LINE_AA)

    if blur_px > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_px)
    if noise > 0:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0.0, noise, image.shape), 0, 255).astype(np.uint8)

    lattice = tuple(lattice_points(path))
    positions = np.array([p.position for p in lattice])
    pixels, depth = project(positions)
    normals = np.array([p.normal for p in lattice])
    facing = np.einsum("ij,ij->i", normals, positions - centre_model) < 0.0
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return TargetView(image, lattice, pixels, facing & inside & (depth > 0))


def look_at(eye_mm, target_mm=(0.0, 0.0, 43.2), up=(0.0, 0.0, 1.0)):
    """A camera pose, as ``(rotation, translation)`` taking model mm to camera mm.

    The default target is the tower's centre of height, so `eye_mm` alone frames
    a view. Handy for generating a calibration sweep without composing matrices
    by hand.
    """
    eye = np.asarray(eye_mm, dtype=float)
    forward = np.asarray(target_mm, dtype=float) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ eye
