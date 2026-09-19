# Live OpenCV HUD

This guide explains how to build a real-time computer vision HUD in OpenCV using `vigilo-stream`, matching the visual presentation of the `deepscreen-viewer` desktop interface.

## HUD features

- **Face detection bounding box**: Crisp green border with corner reticles and confidence badge.
- **5 Facial landmarks**: Distinct colored dots for eyes (green), nose (yellow), and mouth corners (coral).
- **3D Head pose orientation gizmo**: 3-axis Euler angle coordinate frame (Red X, Green Y, Blue Z) projected from the face center.
- **Gaze direction ray**: Magenta ray originating between the eyes pointing along the candidate's line of sight.
- **Prohibited object detection**: Red bounding boxes with label tags for cell phones, laptops, and books.
- **Persistent object hold**: Solves 1 Hz object detector blinking across 30 Hz video frames.
- **Telemetry card**: Dark glassmorphism panel showing FPS, sequence number, face count, and orientation angles.
- **Violation status pills**: Dynamic pills (`NO FACE`, `HEAD TURNED`, `GAZE OFF`, `OBJECT`, `MULTI FACE`) that glow red during active violations.

## Complete OpenCV HUD script

```python
import cv2
import math
import time
import numpy as np
import vigilo_stream

# 1. Initialize pipeline with auto-download
pipe = vigilo_stream.Pipeline(models_dir="models", auto_download=True)
pipe.start("camera:0")

cv2.namedWindow("Vigilo Stream Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vigilo Stream Viewer", 1280, 720)

active_violations = set()
held_objects = []
last_object_time = 0.0

try:
    while pipe.is_running():
        # Zero-copy frame access
        frame = pipe.poll_frame()
        if frame is None:
            continue

        rgb = np.asarray(frame)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Query instantaneous snapshot and new temporal events
        snap = pipe.snapshot()
        for ev in pipe.events():
            if ev.event_type == "ViolationStarted" and ev.violation:
                active_violations.add(ev.violation.kind)
            elif ev.event_type == "ViolationEnded" and ev.violation:
                active_violations.discard(ev.violation.kind)

        if snap:
            # Draw detected faces
            for i, face in enumerate(snap.faces):
                bx = int(face.bbox.x)
                by = int(face.bbox.y)
                bw = int(face.bbox.w)
                bh = int(face.bbox.h)

                # Bounding box & score badge
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (94, 197, 34), 2)
                cv2.putText(
                    img,
                    f"FACE {int(face.score * 100)}%",
                    (bx, max(18, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (140, 255, 100),
                    1,
                    cv2.LINE_AA,
                )

                # 5 Facial landmarks
                palette = [
                    (94, 197, 34),   # Right eye
                    (94, 197, 34),   # Left eye
                    (21, 204, 250),  # Nose
                    (113, 113, 248), # Right mouth
                    (113, 113, 248), # Left mouth
                ]
                for pt, col in zip(face.landmarks, palette):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 4, col, -1, cv2.LINE_AA)

                # 3D Head pose orientation gizmo (Euler projection)
                if i == 0 and snap.head_pose:
                    cx, cy = int(bx + bw * 0.5), int(by + bh * 0.5)
                    s = min(bw, bh) * 0.45
                    rad = math.pi / 180.0
                    y_rad = -snap.head_pose.yaw_deg * rad
                    p_rad = snap.head_pose.pitch_deg * rad
                    r_rad = snap.head_pose.roll_deg * rad

                    cyaw, syaw = math.cos(y_rad), math.sin(y_rad)
                    cpit, spit = math.cos(p_rad), math.sin(p_rad)
                    crol, srol = math.cos(r_rad), math.sin(r_rad)

                    x_end = (int(cx + s * (cyaw * crol)), int(cy + s * (cpit * srol + crol * spit * syaw)))
                    y_end = (int(cx + s * (-cyaw * srol)), int(cy + s * (cpit * crol - spit * syaw * srol)))
                    z_end = (int(cx + s * syaw), int(cy + s * (-cyaw * spit)))

                    cv2.arrowedLine(img, (cx, cy), x_end, (68, 68, 239), 2, tipLength=0.2)
                    cv2.arrowedLine(img, (cx, cy), y_end, (94, 197, 34), 2, tipLength=0.2)
                    cv2.arrowedLine(img, (cx, cy), z_end, (250, 165, 96), 2, tipLength=0.2)

                # Gaze ray
                if i == 0 and snap.gaze and len(face.landmarks) >= 2:
                    ox = int((face.landmarks[0][0] + face.landmarks[1][0]) * 0.5)
                    oy = int((face.landmarks[0][1] + face.landmarks[1][1]) * 0.5)
                    glen = bw * 1.1
                    gdx = -glen * math.sin(snap.gaze.yaw_rad) * math.cos(snap.gaze.pitch_rad)
                    gdy = -glen * math.sin(snap.gaze.pitch_rad)
                    cv2.arrowedLine(img, (ox, oy), (int(ox + gdx), int(oy + gdy)), (252, 171, 240), 2, tipLength=0.15)

            # Prohibited objects with 1.2s persistence hold
            if snap.objects:
                held_objects = snap.objects
                last_object_time = time.time()
            elif "prohibited_object" not in active_violations and (time.time() - last_object_time > 1.2):
                held_objects = []

            for obj in held_objects:
                ox, oy, ow, oh = int(obj.bbox.x), int(obj.bbox.y), int(obj.bbox.w), int(obj.bbox.h)
                cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (68, 68, 239), 2)
                cv2.putText(
                    img,
                    f"{obj.label.upper()} {int(obj.score * 100)}%",
                    (ox, max(18, oy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Vigilo Stream Viewer", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("e"):
            pipe.enrol()

finally:
    pipe.stop()
    cv2.destroyAllWindows()
```

## Why held_objects is needed

The object detection worker runs at 1 Hz to avoid overloading the CPU, while the camera and face workers run at 30 Hz.

When reading `snap.objects` on every frame:
1. `snap.objects` contains detections only on the single frame each second when the 1 Hz inference completes.
2. On the remaining 29 frames of that second, `snap.objects` is empty.
3. If rendered directly, the bounding box blinks for 33 ms and vanishes for 967 ms.

Holding `held_objects` for 1.2 seconds (or while `prohibited_object` is active in `active_violations`) ensures the bounding box renders steadily on screen.
