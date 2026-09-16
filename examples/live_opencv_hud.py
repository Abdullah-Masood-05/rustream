"""
Live OpenCV HUD visualization using vigilo-stream.

Renders real-time webcam video with face bounding boxes, corner reticles,
5 facial landmarks, 3D head pose orientation gizmo, gaze direction rays,
object bounding boxes, telemetry cards, and temporal violation status pills.

Matches the visual presentation of the deepscreen-viewer dashboard.
"""

import argparse
import math
import sys
import time
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import vigilo_stream


def draw_panel(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: Tuple[int, int, int] = (14, 17, 22),
    alpha: float = 0.72,
    border_color: Optional[Tuple[int, int, int]] = (60, 70, 85),
) -> None:
    """Draw a semi-transparent panel with an optional border."""
    img_h, img_w = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    if x1 >= x2 or y1 >= y2:
        return

    overlay = img[y1:y2, x1:x2].copy()
    panel = np.full_like(overlay, bg_color, dtype=np.uint8)
    cv2.addWeighted(panel, alpha, overlay, 1.0 - alpha, 0, overlay)
    img[y1:y2, x1:x2] = overlay

    if border_color is not None:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border_color, 1)


def draw_face_box(img: np.ndarray, x: int, y: int, w: int, h: int, score: float) -> None:
    """Draw face bounding box with corner accent reticles and confidence tag."""
    # Main bounding rectangle (emerald green)
    base_color = (94, 197, 34)
    cv2.rectangle(img, (x, y), (x + w, y + h), base_color, 2)

    # High-tech corner accent brackets
    c_len = max(10, min(24, w // 4, h // 4))
    accent_color = (130, 240, 80)
    # Top-left
    cv2.line(img, (x, y), (x + c_len, y), accent_color, 3)
    cv2.line(img, (x, y), (x, y + c_len), accent_color, 3)
    # Top-right
    cv2.line(img, (x + w, y), (x + w - c_len, y), accent_color, 3)
    cv2.line(img, (x + w, y), (x + w, y + c_len), accent_color, 3)
    # Bottom-left
    cv2.line(img, (x, y + h), (x + c_len, y + h), accent_color, 3)
    cv2.line(img, (x, y + h), (x, y + h - c_len), accent_color, 3)
    # Bottom-right
    cv2.line(img, (x + w, y + h), (x + w - c_len, y + h), accent_color, 3)
    cv2.line(img, (x + w, y + h), (x + w, y + h - c_len), accent_color, 3)

    # Score pill above top-left
    label_text = f"FACE {int(score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    badge_y = max(th + 6, y - 6)
    draw_panel(img, x, badge_y - th - 5, tw + 10, th + 8, bg_color=(20, 30, 24), alpha=0.85, border_color=(94, 197, 34))
    cv2.putText(img, label_text, (x + 5, badge_y - 2), font, font_scale, (140, 255, 100), thickness, cv2.LINE_AA)


def draw_facial_landmarks(img: np.ndarray, landmarks: List[Tuple[float, float]]) -> None:
    """Draw 5 facial keypoints with distinctive colors matching viewer styles."""
    if len(landmarks) < 5:
        return

    # Palette: right eye (green), left eye (green), nose (yellow), right mouth (coral), left mouth (coral)
    colors = [
        (94, 197, 34),    # Right eye
        (94, 197, 34),    # Left eye
        (21, 204, 250),   # Nose
        (113, 113, 248),  # Right mouth corner
        (113, 113, 248),  # Left mouth corner
    ]

    for (lx, ly), col in zip(landmarks[:5], colors):
        pt = (int(round(lx)), int(round(ly)))
        cv2.circle(img, pt, 4, col, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 5, (10, 15, 20), 1, cv2.LINE_AA)


def draw_head_pose_gizmo(img: np.ndarray, x: int, y: int, w: int, h: int, pose) -> None:
    """Draw 3D head pose orientation axes (Red=X, Green=Y, Blue=Z) from face center."""
    cx = int(x + w * 0.5)
    cy = int(y + h * 0.5)
    size = min(w, h) * 0.45

    rad = math.pi / 180.0
    yaw = -pose.yaw_deg * rad  # Negated to match camera subject frame of reference
    pitch = pose.pitch_deg * rad
    roll = pose.roll_deg * rad

    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cpit, spit = math.cos(pitch), math.sin(pitch)
    crol, srol = math.cos(roll), math.sin(roll)

    # 3D Euler projection
    # X axis (Right) - Red
    x_end = (
        int(cx + size * (cyaw * crol)),
        int(cy + size * (cpit * srol + crol * spit * syaw)),
    )
    # Y axis (Down) - Green
    y_end = (
        int(cx + size * (-cyaw * srol)),
        int(cy + size * (cpit * crol - spit * syaw * srol)),
    )
    # Z axis (Forward out of face) - Blue
    z_end = (
        int(cx + size * syaw),
        int(cy + size * (-cyaw * spit)),
    )

    cv2.arrowedLine(img, (cx, cy), x_end, (68, 68, 239), 2, tipLength=0.18, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), y_end, (94, 197, 34), 2, tipLength=0.18, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), z_end, (250, 165, 96), 2, tipLength=0.18, line_type=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, (240, 240, 240), -1, cv2.LINE_AA)


def draw_gaze_ray(img: np.ndarray, x: int, y: int, w: int, h: int, landmarks: List[Tuple[float, float]], gaze) -> None:
    """Draw gaze direction vector originating between the eyes."""
    if len(landmarks) >= 2:
        ox = int(round((landmarks[0][0] + landmarks[1][0]) * 0.5))
        oy = int(round((landmarks[0][1] + landmarks[1][1]) * 0.5))
    else:
        ox = int(x + w * 0.5)
        oy = int(y + h * 0.4)

    length = w * 1.1
    gdx = -length * math.sin(gaze.yaw_rad) * math.cos(gaze.pitch_rad)
    gdy = -length * math.sin(gaze.pitch_rad)
    target = (int(ox + gdx), int(oy + gdy))

    ray_color = (252, 171, 240)  # Magenta/pink
    cv2.circle(img, (ox, oy), 4, ray_color, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, (ox, oy), target, ray_color, 2, tipLength=0.15, line_type=cv2.LINE_AA)


def draw_object_box(img: np.ndarray, obj) -> None:
    """Draw prohibited object bounding box and label tag."""
    x = int(round(obj.bbox.x))
    y = int(round(obj.bbox.y))
    w = int(round(obj.bbox.w))
    h = int(round(obj.bbox.h))

    color = (68, 68, 239)  # Warning red
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    label_text = f"{obj.label.upper()} {int(obj.score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    badge_y = max(th + 6, y - 6)
    draw_panel(img, x, badge_y - th - 5, tw + 10, th + 8, bg_color=(40, 18, 20), alpha=0.88, border_color=color)
    cv2.putText(img, label_text, (x + 5, badge_y - 2), font, font_scale, (255, 200, 200), thickness, cv2.LINE_AA)


def draw_telemetry_hud(
    img: np.ndarray,
    fps: float,
    seq: int,
    face_count: int,
    object_count: int,
    pose,
    gaze,
    is_enrolled: bool,
) -> None:
    """Draw telemetry metrics card on top-left corner."""
    lines = [
        "VIGILO STREAM",
        f"FPS {fps:.1f}   SEQ {seq}",
        f"FACES {face_count}   OBJECTS {object_count}",
    ]

    if pose is not None:
        lines.append(f"POSE yaw {pose.yaw_deg:+.1f} deg  pitch {pose.pitch_deg:+.1f} deg")
    else:
        lines.append("POSE not detected")

    if gaze is not None:
        gaze_yaw_deg = math.degrees(gaze.yaw_rad)
        gaze_pitch_deg = math.degrees(gaze.pitch_rad)
        lines.append(f"GAZE yaw {gaze_yaw_deg:+.1f} deg  pitch {gaze_pitch_deg:+.1f} deg")
    else:
        lines.append("GAZE not detected")

    enrol_text = "ENROLLED" if is_enrolled else "NOT ENROLLED [Press E]"
    lines.append(enrol_text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    line_height = 20
    panel_w = 310
    panel_h = 16 + len(lines) * line_height

    draw_panel(img, 14, 14, panel_w, panel_h, bg_color=(8, 10, 14), alpha=0.78, border_color=(50, 60, 75))

    for i, line in enumerate(lines):
        y_pos = 14 + 18 + (i * line_height)
        if i == 0:
            color = (130, 240, 80)
            cur_font = font
            cur_scale = 0.48
            cur_thick = 1
        elif "ENROLLED" in line:
            color = (100, 220, 120) if is_enrolled else (140, 150, 170)
            cur_font = font
            cur_scale = font_scale
            cur_thick = thickness
        else:
            color = (205, 214, 228)
            cur_font = font
            cur_scale = font_scale
            cur_thick = thickness

        cv2.putText(img, line, (24, y_pos), cur_font, cur_scale, color, cur_thick, cv2.LINE_AA)


def draw_violation_pills(img: np.ndarray, active_violations: Set[str]) -> None:
    """Draw proctoring violation indicator pills on top-right corner."""
    pills = [
        ("NO FACE", "no_face"),
        ("MULTI FACE", "multiple_faces"),
        ("OBJECT", "prohibited_object"),
        ("HEAD TURNED", "head_turned_away"),
        ("GAZE OFF", "gaze_off_screen"),
        ("IDENTITY", "identity_mismatch"),
    ]

    img_w = img.shape[1]
    pill_w = 125
    pill_h = 24
    spacing = 6
    x_pos = img_w - pill_w - 14
    start_y = 14

    # Header label
    cv2.putText(
        img,
        "VIOLATIONS",
        (x_pos, start_y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (124, 135, 151),
        1,
        cv2.LINE_AA,
    )
    start_y += 18

    for label, kind in pills:
        is_active = kind in active_violations
        if is_active:
            # Active glowing red pill
            bg = (40, 40, 220)
            border = (120, 120, 255)
            text_color = (255, 255, 255)
        else:
            # Neutral translucent pill
            bg = (28, 34, 43)
            border = (48, 56, 68)
            text_color = (120, 130, 145)

        draw_panel(img, x_pos, start_y, pill_w, pill_h, bg_color=bg, alpha=0.88, border_color=border)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = x_pos + (pill_w - tw) // 2
        ty = start_y + (pill_h + th) // 2 - 1
        cv2.putText(img, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

        start_y += pill_h + spacing


def main():
    parser = argparse.ArgumentParser(description="Live OpenCV HUD viewer for vigilo-stream")
    parser.add_argument("--source", default="camera:0", help="Video source (e.g. camera:0, file:clip.mp4)")
    parser.add_argument("--models-dir", default="models", help="Directory containing ONNX model weights")
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally (selfie mode)")
    args = parser.parse_args()

    print(f"Starting vigilo-stream pipeline with source: {args.source}")
    print("Controls:")
    print("  [E] Enrol face for identity verification")
    print("  [M] Toggle preview mirror mode")
    print("  [Q] or [ESC] Exit")

    pipe = vigilo_stream.Pipeline(models_dir=args.models_dir, auto_download=True)
    pipe.start(args.source)

    cv2.namedWindow("Vigilo Stream Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vigilo Stream Viewer", 1280, 720)

    fps_counter = 0
    fps = 0.0
    last_fps_time = time.time()
    active_violations: Set[str] = set()
    mirror_mode = args.mirror

    # The object worker runs at 1 Hz to save compute while face detection runs at 30 Hz.
    # We hold the latest detected objects for 1.2 seconds (or while the violation is active)
    # so the bounding box renders steadily without blinking.
    held_objects = []
    last_object_time = 0.0

    try:
        while pipe.is_running():
            # 1. Zero-copy frame polling from Rust pipeline
            frame = pipe.poll_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # Convert RGB24 frame directly to BGR NumPy array for OpenCV display
            rgb_arr = np.asarray(frame)
            img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            if mirror_mode:
                img = cv2.flip(img, 1)

            # 2. Query instantaneous detection signals and fusion events
            snapshot = pipe.snapshot()
            events = pipe.events()

            # Update temporal violation pills
            for ev in events:
                if ev.event_type == "ViolationStarted" and ev.violation is not None:
                    active_violations.add(ev.violation.kind)
                elif ev.event_type == "ViolationEnded" and ev.violation is not None:
                    active_violations.discard(ev.violation.kind)

            # Calculate FPS
            fps_counter += 1
            now = time.time()
            elapsed = now - last_fps_time
            if elapsed >= 0.5:
                fps = fps_counter / elapsed
                fps_counter = 0
                last_fps_time = now

            # 3. Draw detection overlays
            face_count = 0
            head_pose = None
            gaze = None

            if snapshot is not None:
                face_count = len(snapshot.faces)
                head_pose = snapshot.head_pose
                gaze = snapshot.gaze

                # Draw detected faces
                for i, face in enumerate(snapshot.faces):
                    bx = int(round(face.bbox.x))
                    by = int(round(face.bbox.y))
                    bw = int(round(face.bbox.w))
                    bh = int(round(face.bbox.h))

                    if mirror_mode:
                        bx = img.shape[1] - bx - bw

                    draw_face_box(img, bx, by, bw, bh, face.score)

                    # Draw 5 landmarks
                    landmarks = face.landmarks
                    if mirror_mode:
                        landmarks = [(img.shape[1] - lx, ly) for (lx, ly) in landmarks]
                    draw_facial_landmarks(img, landmarks)

                    # Draw 3D head pose orientation gizmo on primary face
                    if i == 0 and head_pose is not None:
                        draw_head_pose_gizmo(img, bx, by, bw, bh, head_pose)

                    # Draw gaze direction ray on primary face
                    if i == 0 and gaze is not None:
                        draw_gaze_ray(img, bx, by, bw, bh, landmarks, gaze)

                # Refresh held objects on detection, hold for 1.2s or during active violation
                if snapshot.objects:
                    held_objects = snapshot.objects
                    last_object_time = time.time()
                elif "prohibited_object" not in active_violations and (time.time() - last_object_time > 1.2):
                    held_objects = []

                # Draw detected prohibited objects (phones, laptops, etc.)
                for obj in held_objects:
                    draw_object_box(img, obj)

            # 4. Draw HUD panels
            draw_telemetry_hud(
                img=img,
                fps=fps,
                seq=frame.seq,
                face_count=face_count,
                object_count=len(held_objects),
                pose=head_pose,
                gaze=gaze,
                is_enrolled=pipe.is_enrolled(),
            )
            draw_violation_pills(img, active_violations)

            # 5. Display frame and handle interactive input
            cv2.imshow("Vigilo Stream Viewer", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("e"), ord("E")):
                pipe.enrol()
                print("Face identity enrollment requested.")
            elif key in (ord("m"), ord("M")):
                mirror_mode = not mirror_mode
                print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
