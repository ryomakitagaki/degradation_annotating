# app.py
import streamlit as st
import io
import zipfile
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

import logic

# --- ヘルパー関数 (エラー回避のため外側に定義) ---
def get_exclusion_mask(image_data, target_w, target_h):
    """canvas の image_data から除外マスクを target サイズで作成する"""
    if image_data is None:
        return None
    # 4チャンネル目(アルファチャンネル)が描画された部分
    alpha = image_data[:, :, 3]
    if alpha.max() == 0:
        return None
    # OpenCVでリサイズ。描画領域を255(白)にする
    mask = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask

# --- UI設定 ---
st.set_page_config(page_title="Crack Batch Labeler", layout="wide")
st.title("🏗️ Degradation Analysis & Annotation Tool")

# --- セッション状態の初期化 ---
if 'file_index' not in st.session_state:
    st.session_state.file_index = 0
if 'results_dict' not in st.session_state:
    st.session_state.results_dict = {}
if 'file_names' not in st.session_state:
    st.session_state.file_names = []
if 'file_bytes_dict' not in st.session_state:
    st.session_state.file_bytes_dict = {}

# --- 基本プロンプト定義 ---
V1="""
写真にうつる建築物の表面を解析し，ひび割れを特定してください。
直線的なタイルやブロックの目地，建材の稜線，塗料の剥がれ部，異種材料の境界部分はひび割れではありません。
建材表面の幾何学的な模様や陰影はひび割れではありません。
特定したひび割れの上に、RGB(255, 0, 0)の純粋な赤色の線を，ひび割れの太さに応じて描画した画像を生成してください。
元の画像と赤色の線のみで構成された画像を返してください。
"""
V2="""
写真に写る建築物の表面を解析し，欠損部や剥離部をすべて特定し、
その範囲にRGB(255, 0, 0)の純粋な赤色を描画した画像を返してください。
"""
V3="""
写真に写る建築物の表面を解析し，エフロレッセンス（白華現象，efflorescence）が見られる領域をすべて特定し、
その範囲にRGB(255, 0, 0)の純粋な赤色で塗りつぶした画像を返してください。
"""
PROMPT_MAP = {
    "Cracks": V1,
    "Chipped/Delaminated": V2,
    "Eflorescence/Other": V3
}

# --- 1. ファイル読み込み（サイドバーより先に実行してfile_namesを確定させる）---
uploaded_files = st.file_uploader("Load images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
current_names = [f.name for f in uploaded_files] if uploaded_files else []

if current_names != st.session_state.file_names:
    st.session_state.file_names = current_names
    st.session_state.file_bytes_dict = {f.name: f.read() for f in uploaded_files} if uploaded_files else {}
    st.session_state.file_index = 0
    st.session_state.results_dict = {}

# --- 2. サイドバー（file_namesが確定した後に描画）---
with st.sidebar:
    st.header("🔑 Setting and model Loading")
    api_key = st.text_input("Gemini API Key", type="password")
    model_id = st.selectbox("Model", ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"])
    prompt_type = st.radio("Degradation Type", ["Cracks", "Chipped/Delaminated", "Eflorescence/Other"])
    min_area = st.number_input("Minimum polygon area (px)", value=10)

    st.divider()
    st.header("📊 Current status")
    if st.session_state.file_names:
        total = len(st.session_state.file_names)
        current = st.session_state.file_index + 1
        st.write(f"Progress: {current} / {total}")
        st.progress(current / total)

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", disabled=(st.session_state.file_index == 0)):
                st.session_state.file_index -= 1
                st.rerun()
        with col_next:
            if st.button("Next ➡️", disabled=(st.session_state.file_index == total - 1)):
                st.session_state.file_index += 1
                st.rerun()

# --- 3. 個別処理エリア ---
if st.session_state.file_names:
    filename = st.session_state.file_names[st.session_state.file_index]
    st.subheader(f"📂 in progress: {filename}")

    col_l, col_r = st.columns([1, 1])
    image_bytes = st.session_state.file_bytes_dict[filename]
    pil_img = Image.open(io.BytesIO(image_bytes))
    w, h = pil_img.size

    with col_l:
        st.image(pil_img, caption="Original Image", use_column_width=True)

        st.write("---")
        st.markdown("#### 🤖 Refinement Prompt")
        refine_key = f"refine_text_{filename}"
        user_refinement = st.text_area(
            "Add instructions for better detection:",
            placeholder="Ex: 'Ignore the vertical tile joints on the right.'",
            key=refine_key
        )

        if st.button("🚀 Analyze / Refine with Gemini", use_container_width=True):
            if not api_key:
                st.error("Please enter API key")
            else:
                with st.spinner("Analyzing..."):
                    final_prompt = PROMPT_MAP[prompt_type]
                    if user_refinement:
                        final_prompt += f"\n\n**Additional instructions:**\n{user_refinement}"

                    traced_data, raw_data = logic.get_gemini_traced_image(api_key, image_bytes, final_prompt, model_id)
                    st.session_state.results_dict[filename] = {"traced_data": traced_data, "raw_data": raw_data}
                    st.success("Analysis completed!")

    with col_r:
        if filename in st.session_state.results_dict and st.session_state.results_dict[filename].get("traced_data"):
            res = st.session_state.results_dict[filename]
            traced_pil = Image.open(io.BytesIO(res["traced_data"]))

            with st.expander("🔍 Check: AI raw output"):
                if res.get("raw_data"):
                    st.image(Image.open(io.BytesIO(res["raw_data"])), caption="Gemini raw output (before color extraction)", use_column_width=True)
                    st.caption("If cracks are visible here but not in the canvas above, the color threshold in `_composite_red_on_original` is filtering them out.")

            st.write("Mark erroneous detection areas (Polygons)")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.5)",
                stroke_width=2,
                background_image=traced_pil,
                update_streamlit=True,
                height=h * (600 / w) if w > 600 else h,
                width=600 if w > 600 else w,
                drawing_mode="polygon",
                key=f"canvas_{filename}",
            )

            # プレビュー処理
            if canvas_result.image_data is not None:
                mask = get_exclusion_mask(canvas_result.image_data, w, h)
                if mask is not None:
                    traced_np = np.array(traced_pil.convert("RGB"))
                    orig_np = np.array(pil_img.convert("RGB"))
                    preview_np = np.where(mask[:, :, np.newaxis] > 0, orig_np, traced_np)
                    st.image(preview_np, caption="Manual Exclusion Preview", use_column_width=True)

            if st.button("✅ Confirm and save", use_container_width=True):
                traced_bytes_to_use = res["traced_data"]

                if canvas_result.image_data is not None:
                    nparr = np.frombuffer(traced_bytes_to_use, np.uint8)
                    traced_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    th, tw = traced_cv.shape[:2]
                    mask = get_exclusion_mask(canvas_result.image_data, tw, th)

                    if mask is not None:
                        orig_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                        orig_resized = cv2.resize(orig_cv, (tw, th))
                        traced_cv[mask > 0] = orig_resized[mask > 0]

                    _, enc = cv2.imencode(".jpg", traced_cv)
                    traced_bytes_to_use = enc.tobytes()

                yolo_txt, vis_img = logic.process_yolo_segmentation(
                    traced_bytes_to_use, w, h, min_area, []
                )

                st.session_state.results_dict[filename].update({
                    "yolo_txt": yolo_txt,
                    "vis_img": vis_img,
                    "completed": True
                })
                st.success(f"Saved: {filename}")

# --- 4. 一括書き出し ---
if st.session_state.results_dict:
    st.divider()
    completed_count = sum(1 for r in st.session_state.results_dict.values() if r.get("completed"))
    if st.button(f"📁 Download ZIP ({completed_count} images)", use_container_width=True):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for fname, data in st.session_state.results_dict.items():
                if data.get("completed"):
                    zf.writestr(f"labels/{Path(fname).stem}.txt", data["yolo_txt"])
                    _, img_enc = cv2.imencode(".jpg", data["vis_img"])
                    zf.writestr(f"visualized/{fname}", img_enc.tobytes())
        st.download_button("🔥 ZIP Download", zip_buffer.getvalue(), "dataset.zip", "application/zip", use_container_width=True)
