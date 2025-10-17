# 硬幣偵測（OpenCV + HoughCircles）

這是一個使用 OpenCV 的硬幣偵測示範。程式以「全域掃描 + ROI 追蹤」的方式進行：每隔 N 幀做一次全域 HoughCircles 偵測，其餘時間僅在上一幀偵測到的圓附近擷取小型 ROI 重新偵測，藉此降低計算量並提升穩定度。視窗會即時顯示偵測框與座標，按下 ESC 離開。

---

## 功能特色

- 全域 + ROI 加速：每隔 `N` 幀跑一次全域偵測，其餘在 ROI 內快速偵測。
- 穩定與效率：在穩定畫面下可減少重複計算、提升 FPS。
- 可調參數：Hough、Canny、半徑範圍與 ROI 邊界可依素材調整。

---

## 環境需求

- Python 3.12 以上（`pyproject.toml` 指定 `>=3.12`）
- 套件：`opencv-python`, `numpy`

安裝方式（擇一）：

```bash
# pip
pip install opencv-python numpy

# 若使用 uv（可參考 uv.lock）
uv pip install opencv-python numpy
```

---

## 專案結構

- `main.py`：主程式（讀取影片、偵測、顯示結果）
- `coin.mp4`：範例影片（請放在與程式同一層）
- `pyproject.toml`：專案與相依套件定義

---

## 執行方式

1) 將欲偵測的影片命名為 `coin.mp4` 並放在與 `main.py` 同層目錄。
2) 執行：

```bash
python main.py
```

操作說明：
- 視窗標題為 `coin`，顯示縮放至 1/2 尺寸的畫面。
- 按下 `ESC` 結束。

若要改用其他影片檔名，請直接修改 `main.py` 內 `cv2.VideoCapture("coin.mp4")` 的路徑字串。

---

## 可調整參數（main.py）

- `GLOBAL_INTERVAL = 10`：每隔多少幀進行一次全域 Hough 偵測。
- `ROI_PAD = 30`：ROI 四周外擴的像素邊界。
- 全域偵測（粗搜）
  - `dp=1.2`, `minDist=30`, `param1=100`, `param2=60`, `minRadius=40`, `maxRadius=80`
- ROI 偵測（細搜）
  - `dp=1.2`, `minDist=20`, `param1=100`, `param2=50`, `minRadius=max(20, r-20)`, `maxRadius=r+20`

調參建議：
- `param2` 越小越容易出現偽陽性，越大則越嚴格。
- `minRadius/maxRadius` 請符合實際目標大小，過大或過小都會影響檢出率。
- 畫面晃動或縮放差異較大時，可調大 `ROI_PAD` 或縮短 `GLOBAL_INTERVAL`。

---

## 偵測流程摘要

1. 讀取一幀影像，轉灰階並以高斯模糊抑制雜訊。
2. 決定是否進行全域偵測（每 `GLOBAL_INTERVAL` 幀一次，或無前次結果時）。
3. 若不全域偵測，則依照上一幀的圓心與半徑在附近擷取 ROI，於 ROI 內執行 HoughCircles。
4. 合併偵測結果，於原圖繪製紅色方框與 `(x, y)` 座標。
5. 以 1/2 尺寸顯示畫面，直到按下 ESC 結束。

---

## 常見問題

- 無法開啟影片：確認檔案路徑正確、權限允許，且編碼/格式為 OpenCV 可讀。
- 漏檢或誤檢：微調 `param2`、`minDist`、半徑範圍，或提高影像品質（亮度、對焦）。
- 效能不足：加大 `GLOBAL_INTERVAL`、縮小 ROI（降低 `ROI_PAD`）或降低輸入解析度。

---

## 版權與授權

僅供教學與示範使用，未附帶授權條款。若需正式授權或商用，請自行補充授權資訊。

