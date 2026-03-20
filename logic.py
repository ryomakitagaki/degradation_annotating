# logic.py
import cv2
import numpy as np
from google import genai
from google.genai import types

def get_gemini_traced_image(api_key, image_bytes, prompt, model_id):
    """Gemini APIを呼び出して、劣化箇所に赤を描画した画像データを取得する"""
    # ⚠️ (ここは以前と変更なし)
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                raw_bytes = part.inline_data.data
                composite_bytes = _composite_red_on_original(image_bytes, raw_bytes)
                return composite_bytes, raw_bytes
        return None, None
    except Exception as e:
        raise Exception(f"APIエラー: {e}")

def _extract_red_mask(bgr_img, saturation_threshold=150):
    """HSV色空間で赤系の色を検出する（JPEG圧縮による色ズレに強い）"""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # 赤はHSVで色相が0〜10と170〜180の2範囲に分かれる
    # 彩度閾値を高めにして，JPEGアーティファクトによる太線化を抑制
    mask1 = cv2.inRange(hsv, np.array([0,   saturation_threshold, 80]), np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, saturation_threshold, 80]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)

def _composite_red_on_original(original_bytes, traced_bytes):
    orig = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR)
    traced = cv2.imdecode(np.frombuffer(traced_bytes, np.uint8), cv2.IMREAD_COLOR)
    if traced.shape[:2] != orig.shape[:2]:
        traced = cv2.resize(traced, (orig.shape[1], orig.shape[0]))

    # 差分ベースの赤検出:
    # 元画像と比較して「Rチャンネルが増加 かつ G/Bが減少」したピクセル = Geminiが描いた赤線
    orig_f = orig.astype(np.float32)
    traced_f = traced.astype(np.float32)
    r_increased = (traced_f[:, :, 2] - orig_f[:, :, 2]) > 40   # Rが40以上増加
    g_decreased = (orig_f[:, :, 1] - traced_f[:, :, 1]) > 10   # Gが減少
    b_decreased = (orig_f[:, :, 0] - traced_f[:, :, 0]) > 10   # Bが減少
    traced_is_red = traced_f[:, :, 2] > 150                     # tracedのR値が高い

    red_mask = (r_increased & g_decreased & b_decreased & traced_is_red).astype(np.uint8) * 255

    # 細線化: erosionでJPEG圧縮による滲みを除去してからdilateで少し膨張
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    red_mask = cv2.erode(red_mask, kernel, iterations=1)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.dilate(red_mask, kernel_dilate, iterations=1)

    result = orig.copy()
    result[red_mask > 0] = [0, 0, 255]
    _, enc = cv2.imencode(".jpg", result)
    return enc.tobytes()


def process_yolo_segmentation(traced_bytes, original_width, original_height, min_area_px=10, exclusion_rects=None):
    """赤描画画像からYOLO用テキストと可視化画像を生成する (除外領域処理付き)"""
    
    # バイト列をOpenCV形式に変換
    nparr = np.frombuffer(traced_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 赤色領域の抽出（HSV）
    mask = _extract_red_mask(img)

    # ⚠️ 【新機能】 除外領域（矩形リスト）の処理
    # exclusion_rects: [{'left': x, 'top': y, 'width': w, 'height': h}, ...]
    if exclusion_rects:
        for rect in exclusion_rects:
            # 座標を取得
            x = int(rect['left'])
            y = int(rect['top'])
            w = int(rect['width'])
            h = int(rect['height'])
            
            # マスク画像の指定された矩形領域を黒（0：検出なし）で塗りつぶす
            # 確実に削除するため、塗りつぶし (thickness=-1)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
            
            # 可視化画像の方にも、除外領域をグレーの矩形で描画（確認用）
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)

    # 以降は以前と同じ処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        coords = []
        for pt in approx:
            nx = max(0.0, min(1.0, pt[0][0] / original_width))
            ny = max(0.0, min(1.0, pt[0][1] / original_height))
            coords.append(f"{nx:.6f} {ny:.6f}")
        yolo_lines.append(f"0 {' '.join(coords)}")

    return "\n".join(yolo_lines), img