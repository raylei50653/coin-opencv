#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可讀性優先的圓形（硬幣）偵測：全域 Hough + ROI Hough
- 清楚的常數集中管理
- 小函式拆分（I/O、偵測、繪圖、FPS）
- 型別註記 + 文件字串
"""

from __future__ import annotations
import os
from collections import deque
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ========= 參數集中管理 =========
VIDEO_PATH: str = "coin.mp4"

# Hough 相關
DP: float = 1.0             # 全域 Hough 的累加器解析度
MIN_DIST: int = 20          # 圓心最小距離
MIN_R: int = 15             # 全域半徑下限
MAX_R: int = 80             # 全域半徑上限
P1: int = 120               # Canny 高閾值（低閾值 = 0.5*P1）
P2: int = 40                # Hough 投票閾值（越小越敏感）

# ROI 模式
DETECT_INTERVAL: int = 6    # 每 N 幀做一次全域偵測
ROI_PAD_FACTOR: float = 0.5  # ROI 外擴比例（相對半徑）

# 顯示
WINDOW_NAME: str = "Video"
FPS_SMOOTH_WIN: int = 30


# ========= 小工具 =========
def clamp(v: int, lo: int, hi: int) -> int:
    """限制整數 v 落在 [lo, hi] 區間內。"""
    return max(lo, min(hi, v))


def open_video(path: str) -> cv2.VideoCapture:
    """開啟影片/來源並檢查狀態。"""
    if not os.path.exists(path):
        raise SystemExit(f"找不到影片檔案：{path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{path}")
    return cap


def read_video_info(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    """讀取影片的寬、高、FPS。"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"影片資訊: {width}x{height}, {fps:.1f} FPS")
    return width, height, fps


def create_window(width: int, height: int) -> None:
    """建立可調整大小的視窗。"""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, max(320, width // 2), max(240, height // 2))


def smooth_fps(t_prev: int, fps_deque: deque) -> Tuple[float, int]:
    """
    使用 OpenCV 計時器計算影格間隔，並回傳平滑後的 FPS 與新的時間戳。
    """
    t_now = cv2.getTickCount()
    dt = (t_now - t_prev) / cv2.getTickFrequency()
    if dt > 1e-6:
        fps_deque.append(1.0 / dt)
    avg_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0.0
    return avg_fps, t_now


# ========= 偵測（全域 / ROI）=========
def hough_on_gray(blurred_gray: np.ndarray,
                  min_r: int, max_r: int) -> Optional[np.ndarray]:
    """
    在整張模糊灰階圖上執行 Hough 圓偵測。
    回傳格式同 cv2.HoughCircles（或 None）。
    """
    return cv2.HoughCircles(
        blurred_gray,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=P1,
        param2=P2,
        minRadius=min_r,
        maxRadius=max_r
    )


def hough_on_roi(blurred_gray: np.ndarray,
                 px: int, py: int, pr: int,
                 width: int, height: int) -> List[Tuple[int, int, int]]:
    """
    在上一幀已知圓 (px, py, pr) 的周邊 ROI 內進行 Hough 圓偵測。
    回傳全圖座標系的圓列表 [(x, y, r), ...]。
    """
    # 1) 建立 ROI 區域（外擴一定比例）
    pad = int(pr * (1.0 + ROI_PAD_FACTOR))
    x0, y0 = clamp(px - pad, 0, width - 1), clamp(py - pad, 0, height - 1)
    x1, y1 = clamp(px + pad, 0, width - 1), clamp(py + pad, 0, height - 1)
    if x1 <= x0 or y1 <= y0:
        return []

    roi = blurred_gray[y0:y1, x0:x1]

    # 2) ROI 內的半徑與 minDist 自動縮放（比全域更貼近上一幀大小）
    min_r = clamp(int(pr * 0.7), 1, MAX_R)
    max_r = clamp(int(pr * 1.3), min_r + 1, MAX_R)
    roi_min_dist = max(8, int(pr * 0.8))

    # 3) 執行 ROI Hough（dp=1.0 以提高精度）
    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=roi_min_dist,
        param1=P1,
        param2=P2,
        minRadius=min_r,
        maxRadius=max_r
    )

    # 4) 轉回全圖座標
    found: List[Tuple[int, int, int]] = []
    if circles is not None:
        circles = np.around(circles).astype(np.uint16)
        assert isinstance(circles, np.ndarray)
        for (rx, ry, rr) in circles[0, :]:
            found.append((int(rx + x0), int(ry + y0), int(rr)))
    return found


def detect_circles(frame_idx: int,
                   prev_circles: List[Tuple[int, int, int]],
                   blurred_gray: np.ndarray,
                   width: int, height: int) -> Optional[np.ndarray]:
    """
    決定執行全域偵測或 ROI 偵測，並將結果統一成 HoughCircles 的格式。
    """
    # 每隔 DETECT_INTERVAL 幀做一次全域偵測，或第一幀 / 沒有前次結果時
    is_global = (frame_idx % DETECT_INTERVAL == 1) or (len(prev_circles) == 0)

    if is_global:
        return hough_on_gray(blurred_gray, MIN_R, MAX_R)

    # ROI 模式：聚合所有 ROI 的偵測結果
    rois_found: List[Tuple[int, int, int]] = []
    for (px, py, pr) in prev_circles:
        rois_found.extend(hough_on_roi(blurred_gray, px, py, pr, width, height))

    return np.array([rois_found], dtype=np.float32) if rois_found else None


# ========= 視覺化 =========
def draw_detections(vis: np.ndarray,
                    circles: Optional[np.ndarray],
                    width: int, height: int) -> List[Tuple[int, int, int]]:
    """
    把偵測結果畫在畫面上，並回傳當前幀的圓清單（供下幀當作 prev_circles）。
    """
    if circles is None:
        return []
    circles = np.around(circles).astype(np.uint16)
    current: List[Tuple[int, int, int]] = []

    for (x, y, r) in circles[0, :]:
        r = int(r)
        x0, y0 = clamp(x - r, 0, width - 1), clamp(y - r, 0, height - 1)
        x1, y1 = clamp(x + r, 0, width - 1), clamp(y + r, 0, height - 1)

        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
        cv2.putText(vis, f"{x},{y}",
                    (max(0, x0), max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2, cv2.LINE_AA)

        current.append((int(x), int(y), r))

    return current


# ========= 主程式 =========
def main() -> None:
    # 1) 讀取來源與資訊
    cap = open_video(VIDEO_PATH)
    width, height, _ = read_video_info(cap)

    # 2) 視窗、FPS 緩衝
    create_window(width, height)
    fps_deque: deque = deque(maxlen=FPS_SMOOTH_WIN)
    t_prev = cv2.getTickCount()

    prev_circles: List[Tuple[int, int, int]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("影片結束")
                break

            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # 3) 偵測
            circles = detect_circles(frame_idx, prev_circles, blurred, width, height)

            # 4) 繪圖 + FPS
            vis = frame.copy()
            prev_circles = draw_detections(vis, circles, width, height)

            fps, t_prev = smooth_fps(t_prev, fps_deque)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)

            # 5) 按鍵控制
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q / ESC
                print("影片中斷")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("處理完成！")


if __name__ == "__main__":
    main()