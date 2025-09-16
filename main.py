import cv2
import math
import time
import mediapipe as mp
import numpy as np

URL = "http://172.20.91.210:4747/video"

# Tunables
EAR_STRICT = 0.19
IRIS_STRICT = 0.55
CLOSED_MIN_DURATION = 3.0
SHOW_FPS = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE_RING  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT = {"corners": (33, 133), "uppers": (159, 160), "lowers": (145, 144), "iris": LEFT_IRIS}
RIGHT = {"corners": (362, 263), "uppers": (386, 385), "lowers": (374, 380), "iris": RIGHT_IRIS}

# --- Helpers ---
def dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
def px(lm, w, h, i):
    p = lm[i]; return (int(p.x * w), int(p.y * h))

def eye_measures(landmarks, w, h, spec):
    c1, c2 = px(landmarks, w, h, spec["corners"][0]), px(landmarks, w, h, spec["corners"][1])
    u1, u2 = px(landmarks, w, h, spec["uppers"][0]),  px(landmarks, w, h, spec["uppers"][1])
    l1, l2 = px(landmarks, w, h, spec["lowers"][0]),  px(landmarks, w, h, spec["lowers"][1])
    horiz = dist(c1, c2) + 1e-6
    opening = (dist(u1, l1) + dist(u2, l2)) / 2.0
    ear_like = opening / horiz
    iris_pts = [px(landmarks, w, h, i) for i in spec["iris"]]
    icx, icy = iris_pts[0]
    r = np.mean([dist((icx, icy), p) for p in iris_pts[1:]]) if len(iris_pts) > 1 else 1.0
    iris_diam = 2.0 * max(r, 1.0)
    opening_vs_iris = opening / (iris_diam + 1e-6)
    return ear_like, opening_vs_iris, (c1, c2, u1, u2, l1, l2), iris_pts

def draw_panel(img, x, y, w, h, color=(0,0,0), alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_progress_bar(img, x, y, w, h, progress, bg=(40,40,40), fg=(0,180,255)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    pw = int(w * max(0.0, min(1.0, progress)))
    cv2.rectangle(img, (x, y), (x+pw, y+h), fg, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 1)

def put_text(img, text, org, scale=0.8, color=(255,255,255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_COMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_polyline(img, pts, closed=True, color=(0,255,255), thickness=2):
    pts = np.array(pts, dtype=np.int32).reshape((-1,1,2))
    cv2.polylines(img, [pts], closed, color, thickness, cv2.LINE_AA)

def draw_translucent_fill(img, pts, color=(0,255,255), alpha=0.25):
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def smoothe(prev, new, a=0.5):
    return new if prev is None else (a*new + (1-a)*prev)

# --- Compact HUD helpers ---
def text_size(text, scale=0.8, thick=1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, scale, thick)
    return w, h, base

def draw_compact_top_bar(frame, fully_closed, alert_active, ear_val, iris_val, fps_val):
    h, w = frame.shape[:2]

    # Decide bar color
    if alert_active:      bar_color = (0, 0, 255)     # RED
    elif fully_closed:    bar_color = (0, 255, 255)   # CYAN
    else:                 bar_color = (0, 255, 0)     # GREEN

    # Content strings
    status = ("Status: ALERT - Eyes Fully Closed ≥ 3s" if alert_active
              else "Status: Eyes Fully Closed" if fully_closed
              else "Status: Normal")

    right_lines = [
        f"EAR {ear_val:.3f}" if ear_val is not None else "EAR --",
        f"Iris {iris_val:.2f}" if iris_val is not None else "Iris --",
    ]
    if SHOW_FPS:
        right_lines.append(f"FPS {fps_val:.1f}")

    # Measure texts to compute bar height & layout
    left_scale, right_scale = 0.95, 0.8
    left_w, left_h, _ = text_size(status, left_scale, 2)

    right_sizes = [text_size(s, right_scale, 1) for s in right_lines]
    right_w = max(ws for (ws, _, _) in right_sizes)
    right_h_sum = sum(hs for (_, hs, _) in right_sizes) + (len(right_sizes)-1)*6

    pad_y, pad_x = 10, 16
    bar_h = max(left_h + 2*pad_y, right_h_sum + 2*pad_y)
    draw_panel(frame, 0, 0, w, bar_h, color=bar_color, alpha=0.35)

    # Left: status (vertically centered)
    left_y = (bar_h + left_h)//2 - 3
    put_text(frame, status, (pad_x, left_y), scale=left_scale, color=(255,255,255), thick=2)

    # Right: stacked metrics (top-right, inside bar)
    x_right = w - pad_x - right_w
    y_cursor = pad_y + right_sizes[0][1]  # start with first line height
    for idx, line in enumerate(right_lines):
        put_text(frame, line, (x_right, y_cursor), scale=right_scale, color=(255,255,255), thick=1)
        if idx < len(right_lines)-1:
            y_cursor += right_sizes[idx+1][1] + 6  # next line height + spacing

    return bar_h  # return height to avoid overlaps below

def draw_bottom_progress(frame, elapsed_closed, CLOSED_MIN_DURATION, alert_active):
    h, w = frame.shape[:2]
    band_h = 54
    draw_panel(frame, 0, h - band_h, w, band_h, color=(0,0,0), alpha=0.35)

    # Centered progress bar
    bar_w = max(300, int(w*0.55))
    bar_h = 18
    bar_x = (w - bar_w)//2
    bar_y = h - band_h//2 - bar_h//2
    progress = (elapsed_closed / CLOSED_MIN_DURATION) if CLOSED_MIN_DURATION > 0 else 0.0
    draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h, progress,
                      bg=(60,60,60),
                      fg=((0,180,255) if not alert_active else (0,0,255)))

    # Label (centered above bar)
    label = f"Fully closed: {elapsed_closed:.1f}s / {CLOSED_MIN_DURATION:.0f}s"
    lw, lh, _ = text_size(label, 0.8, 2)
    put_text(frame, label, ((w - lw)//2, bar_y - 8), scale=0.8, color=(255,255,255), thick=2)

    # Hint (bottom-left inside band)
    put_text(frame, "Press 'Q' to quit", (16, h - 14), scale=0.6, color=(255,255,255), thick=1)

# --- Video ---
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    raise RuntimeError(
        f"Could not open stream: {URL}\n"
        f"- Check DroidCam Wi-Fi IP/Port and ensure both devices are on the same network.\n"
        f"- Test in a browser first; if it works there, it should work here."
    )

closed_since = None
last_time = time.time()
frames = 0
fps_val = 0.0
alert_active = False

L_ear_s = R_ear_s = None
L_iris_s = R_iris_s = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from stream.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    fully_closed = False
    elapsed_closed = 0.0
    ear_avg_display = None
    iris_ratio_avg_display = None

    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark

        xs = [int(p.x * w) for p in lms]
        ys = [int(p.y * h) for p in lms]
        x1, y1, x2, y2 = max(min(xs), 0), max(min(ys), 0), min(max(xs), w-1), min(max(ys), h-1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
        put_text(frame, "Tracking", (x1 + 6, max(72, y1 - 8)), scale=0.7, color=(120,220,120), thick=2)  # avoid top bar overlap

        L_ear, L_iris_ratio, L_keypts, L_iris_pts = eye_measures(lms, w, h, LEFT)
        R_ear, R_iris_ratio, R_keypts, R_iris_pts = eye_measures(lms, w, h, RIGHT)

        L_ear_s  = smoothe(L_ear_s,  L_ear,  a=0.5)
        R_ear_s  = smoothe(R_ear_s,  R_ear,  a=0.5)
        L_iris_s = smoothe(L_iris_s, L_iris_ratio, a=0.5)
        R_iris_s = smoothe(R_iris_s, R_iris_ratio, a=0.5)

        ear_avg_display = (L_ear_s + R_ear_s) / 2.0
        iris_ratio_avg_display = (L_iris_s + R_iris_s) / 2.0

        L_closed = (L_ear_s is not None and L_iris_s is not None and L_ear_s < EAR_STRICT and L_iris_s < IRIS_STRICT)
        R_closed = (R_ear_s is not None and R_iris_s is not None and R_ear_s < EAR_STRICT and R_iris_s < IRIS_STRICT)
        fully_closed = L_closed and R_closed

        def ring_points(indices):
            return [(int(lms[i].x*w), int(lms[i].y*h)) for i in indices]
        left_ring_pts  = ring_points(LEFT_EYE_RING)
        right_ring_pts = ring_points(RIGHT_EYE_RING)
        draw_translucent_fill(frame, left_ring_pts,  color=(0,200,255), alpha=0.22)
        draw_translucent_fill(frame, right_ring_pts, color=(0,200,255), alpha=0.22)
        draw_polyline(frame, left_ring_pts,  True, color=(0,255,255), thickness=2)
        draw_polyline(frame, right_ring_pts, True, color=(0,255,255), thickness=2)

        for (c1, c2, u1, u2, l1, l2) in [L_keypts, R_keypts]:
            for p in (u1, l1, u2, l2):
                cv2.circle(frame, p, 3, (255,255,255), -1, cv2.LINE_AA)
        for ip in [L_iris_pts[0], R_iris_pts[0]]:
            cv2.circle(frame, ip, 3, (0,255,0), -1, cv2.LINE_AA)
    else:
        closed_since = None
        elapsed_closed = 0.0

    # Closed timer & alert logic
    if result.multi_face_landmarks and fully_closed:
        if closed_since is None:
            closed_since = time.time()
        elapsed_closed = time.time() - closed_since
    else:
        closed_since = None
        elapsed_closed = 0.0
    alert_active = elapsed_closed >= CLOSED_MIN_DURATION

    # FPS
    if SHOW_FPS:
        frames += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps_val = frames / (now - last_time)
            last_time, frames = now, 0

    # --- COMPACT HUD ---
    top_h = draw_compact_top_bar(
        frame,
        fully_closed=fully_closed,
        alert_active=alert_active,
        ear_val=ear_avg_display,
        iris_val=iris_ratio_avg_display,
        fps_val=fps_val
    )

    # Bottom progress band (centered)
    draw_bottom_progress(frame, elapsed_closed, CLOSED_MIN_DURATION, alert_active)

    # Red border when alert
    if alert_active:
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 6)

    cv2.imshow("DroidCam - Drowsiness Monitor (Strict Close)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
