from PIL import Image
import pytesseract
import cv2
import numpy as np

# Tesseractのインストールパスを指定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 画像ファイルのパスを指定
image_path = r"\S__161243140_0.jpg"

# 画像を開く（OpenCV形式で読み込み）
img = cv2.imread(image_path)

# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ノイズ除去（GaussianBlurで平滑化）
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 二値化（閾値処理）
_, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 画像を表示（オプション：処理後の画像確認）
# cv2.imshow("Processed Image", binary_image)
# cv2.waitKey(0)

# PIL形式に変換
pil_img = Image.fromarray(binary_image)

# 画像から日本語テキストを抽出
text = pytesseract.image_to_string(pil_img, lang="jpn")

# 抽出されたテキストを表示
print(text)
