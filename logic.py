# logic.py
import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image
import io

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
                return part.inline_data.data
        return None
    except Exception as e:
        raise Exception(f"APIエラー: {e}")

def process_yolo_segmentation(traced_bytes, original_width, original_height, min_area_px=10, exclusion_rects=None):
    """赤描画画像からYOLO用テキストと可視化画像を生成する (除外領域処理付き)"""
    
    # バイト列をOpenCV形式に変換
    nparr = np.frombuffer(traced_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 赤色領域の抽出（BGR）
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([50, 50, 255])
    mask = cv2.inRange(img, lower_red, upper_red)

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