# 標準庫
import os
import time
from collections import deque
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# 第三方套件
import cv2
import numpy as np


video_path = "coin.mp4"

if not os.path.exists(video_path):
    raise SystemExit(f"找不到影片檔案：{video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"無法開啟影片：{video_path}")

# 影片參數
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"影片資訊: {width}x{height}, {fps:.1f} FPS")

fps_deque = deque(maxlen=30)
t = cv2.getTickCount()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", max(320, width // 2), max(240, height // 2))

# —— 關鍵參數（可依畫面微調）——
DP = 1             # 累加器解析度比（>1 會降解析度以加速）
MIN_DIST = 20            # 兩圓心最小距離（像素）
MIN_R = 15            # 半徑下限（像素）
MAX_R = 80            # 半徑上限（像素）
P1 = 120           # 內部 Canny 高閾值（低閾值=0.5*P1）
P2 = 40            # 投票閾值，越小越敏感（建議 25–40 起手）

DETECT_INTERVAL = 6      # 每 N 幀做一次「全局偵測」
ROI_PAD_FACTOR = 0.35    # ROI 邊界外擴比例（相對半徑）

prev_circles = []         # 上一輪的 [(x, y, r)]
frame_idx = 0


class ROIHoughDetector:
    def __init__(self, worker_cap=None, min_rois_for_threading=2):
        self.worker_cap = worker_cap or multiprocessing.cpu_count()
        self.min_rois_for_threading = min_rois_for_threading
        self.executor = ThreadPoolExecutor(max_workers=self.worker_cap)

    def close(self):
        self.executor.shutdown(wait=True)

    # 也可做成 context manager
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

    def detect(self, prev_circles, blurred, width, height):
        if not prev_circles:
            return None

        if len(prev_circles) < self.min_rois_for_threading:
            rois_found = []
            for (px, py, pr) in prev_circles:
                rois_found.extend(hough_on_roi(
                    blurred, px, py, pr, width, height))
            return np.array([rois_found], dtype=np.float32) if rois_found else None

        rois_found = []
        futures = [
            self.executor.submit(hough_on_roi, blurred,
                                 px, py, pr, width, height)
            for (px, py, pr) in prev_circles
        ]
        for f in as_completed(futures):
            try:
                result = f.result()
                if result:
                    rois_found.extend(result)
            except Exception as e:
                print(f"[警告] ROI 偵測執行緒錯誤：{e}")
        return np.array([rois_found], dtype=np.float32) if rois_found else None


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def hough_on_gray(gray_blur, min_r, max_r):
    return cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=P1,
        param2=P2,
        minRadius=min_r,
        maxRadius=max_r
    )


def hough_on_roi(blurred_img, px, py, pr, width, height):
    """
    在單一 ROI 內做局部 Hough 圓偵測。
    參數：
        blurred_img : 灰階模糊影像
        px, py, pr  : 上一輪圓心與半徑
        width, height : 原圖尺寸，用來 clamp
    回傳：
        list of (gx, gy, rr)  —— 還原到全圖座標的圓
    """
    pad = int(pr * (1.0 + ROI_PAD_FACTOR))
    x0 = clamp(px - pad, 0, width - 1)
    y0 = clamp(py - pad, 0, height - 1)
    x1 = clamp(px + pad, 0, width - 1)
    y1 = clamp(py + pad, 0, height - 1)
    if x1 <= x0 or y1 <= y0:
        return []

    roi = blurred_img[y0:y1, x0:x1]

    # ROI 專用參數
    min_r = clamp(int(pr * 0.7), 1, MAX_R)
    max_r = clamp(int(pr * 1.3), min_r + 1, MAX_R)
    roi_minDist = max(8, int(pr * 0.8))
    roi_dp = 1.0  # 小 ROI 精度高一點

    # 執行 Hough
    c = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=roi_dp, minDist=roi_minDist,
        param1=P1, param2=P2, minRadius=min_r, maxRadius=max_r
    )

    # 將 ROI 座標轉回全圖
    found = []
    if c is not None:
        c = np.uint16(np.around(c))
        for (rx, ry, rr) in c[0, :]:
            gx, gy = rx + x0, ry + y0
            found.append((gx, gy, rr))
    return found


def fps_smooth(t):
    t1 = cv2.getTickCount()
    dt = (t1 - t) / cv2.getTickFrequency()
    if dt > 1e-6:
        fps_deque.append(1.0 / dt)
    smooth_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0.0
    return smooth_fps, t1   # 回傳新的 t1


with ROIHoughDetector(worker_cap=8, min_rois_for_threading=2) as roi_det:
    frame_idx = 0
    t = cv2.getTickCount()

    while True:
        ret, frame = cap.read()            # ← 移進來每圈讀一張
        if not ret:
            print("影片結束或讀取失敗")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # === 全域 / ROI 偵測 ===
        if (frame_idx % DETECT_INTERVAL == 1) or (len(prev_circles) == 0):
            circles = hough_on_gray(blurred, MIN_R, MAX_R)
        else:
            circles = roi_det.detect(prev_circles, blurred, width, height)

        # === 繪圖 ===
        vis = frame.copy()
        new_prev = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                r = int(r)
                x0 = clamp(x - r, 0, width - 1)
                y0 = clamp(y - r, 0, height - 1)
                x1 = clamp(x + r, 0, width - 1)
                y1 = clamp(y + r, 0, height - 1)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
                cv2.putText(vis, f"{x},{y}", (max(0, x0), max(20, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2, cv2.LINE_AA)
                new_prev.append((x, y, r))

        smooth_fps, t = fps_smooth(t)
        cv2.putText(vis, f"FPS: {smooth_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Video", vis)
        prev_circles = new_prev

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC 離開
            print("影片中斷")
            break

cap.release()
cv2.destroyAllWindows()
print("處理完成！")
