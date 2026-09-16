import sys
from pathlib import Path
import pytest

# Gracefully skip if OpenCV is not installed in the test environment
cv2 = pytest.importorskip("cv2", reason="OpenCV is not installed")

# Add project root to sys.path so examples can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import vigilo_stream
from examples.live_opencv_hud import (
    draw_face_box,
    draw_facial_landmarks,
    draw_gaze_ray,
    draw_head_pose_gizmo,
    draw_object_box,
    draw_panel,
    draw_telemetry_hud,
    draw_violation_pills,
)


def test_drawing_functions_on_blank_frame():
    # 720p frame canvas
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Semi-transparent panel
    draw_panel(img, 10, 10, 200, 100)

    # 2. Face bounding box & score
    draw_face_box(img, 300, 150, 200, 250, 0.96)

    # 3. 5 Facial landmarks
    landmarks = [(340.0, 210.0), (440.0, 210.0), (390.0, 260.0), (350.0, 320.0), (430.0, 320.0)]
    draw_facial_landmarks(img, landmarks)

    # 4. 3D Head pose orientation axes
    pose = vigilo_stream.HeadPose(yaw_deg=10.0, pitch_deg=-5.0, roll_deg=2.0)
    draw_head_pose_gizmo(img, 300, 150, 200, 250, pose)

    # 5. Gaze direction ray
    gaze = vigilo_stream.Gaze(yaw_rad=-0.12, pitch_rad=0.06)
    draw_gaze_ray(img, 300, 150, 200, 250, landmarks, gaze)

    # 6. Object box
    bbox = vigilo_stream.BBox(600.0, 300.0, 120.0, 180.0)
    obj = vigilo_stream.ObjectDetection("cell phone", 0.91, bbox)
    draw_object_box(img, obj)

    # 7. Telemetry HUD & violation status pills
    draw_telemetry_hud(
        img=img,
        fps=30.0,
        seq=50,
        face_count=1,
        object_count=1,
        pose=pose,
        gaze=gaze,
        is_enrolled=True,
    )
    draw_violation_pills(img, {"head_turned_away", "prohibited_object"})

    assert img.shape == (720, 1280, 3)
    assert img.dtype == np.uint8
    # Canvas should have drawn non-zero pixel values
    assert np.any(img > 0)
