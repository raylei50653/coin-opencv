import cv2
import numpy as np

cap = cv2.VideoCapture("coin.mp4")
if not cap.isOpened():
    raise SystemExit("無法開啟影片")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

GLOBAL_INTERVAL = 10   # 每 N 幀全域偵測
ROI_PAD = 30           # 區域外擴像素
last_circles = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = None

    use_global = (frame_idx % GLOBAL_INTERVAL == 1) or (last_circles is None)
    if use_global:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=100, param2=60,
            minRadius=40, maxRadius=80
        )
    else:
        last = np.rint(last_circles).astype(np.int32).reshape(-1, 3)
        rois_found = []
        for x, y, r in last:
            x1, y1 = max(0, x - r - ROI_PAD), max(0, y - r - ROI_PAD)
            x2, y2 = min(w, x + r + ROI_PAD), min(h, y + r + ROI_PAD)
            roi = blur[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            rc = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=20,
                param1=100, param2=50,
                minRadius=max(20, r - 20),
                maxRadius=r + 20
            )
            if rc is None:
                continue
            rc = np.rint(rc[0]).astype(np.int32)
            rc[:, 0] += x1
            rc[:, 1] += y1
            rois_found.extend(rc.tolist())
        circles = np.array([rois_found], dtype=np.float32) if rois_found else None

    if circles is not None:
        rounded = np.rint(circles[0]).astype(np.int32)
        for x, y, r in rounded:
            cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 0, 255), 4)
            cv2.putText(frame, f"{x},{y}",
                        (max(0, x - r), max(20, y - r - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
        last_circles = rounded[np.newaxis, :, :].astype(np.float32)

    cv2.imshow("coin", cv2.resize(frame, (w // 2, h // 2)))
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
