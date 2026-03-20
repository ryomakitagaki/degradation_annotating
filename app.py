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

# --- UI設定 ---
st.set_page_config(page_title="Crack Batch Labeler", layout="wide")
st.title("🏗️ ひび割れ一括解析・アノテーションツール")

# --- セッション状態の初期化 ---
if 'file_index' not in st.session_state:
    st.session_state.file_index = 0             # 現在表示中のファイル番号
if 'results_dict' not in st.session_state:
    st.session_state.results_dict = {}         # 解析結果の保存用 {ファイル名: {yolo_txt, vis_img, traced_data}}
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []  # アップロードされたファイル群

# --- サイドバー ---
with st.sidebar:
    st.header("🔑 認証・設定")
    api_key = st.text_input("Gemini API Key", type="password")
    model_id = st.selectbox("モデル", ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"])
    prompt_type = st.radio("検出種類", ["ひび割れ", "欠損・剥離", "その他"])
    min_area = st.number_input("最小ポリゴン面積(px)", value=10)

    st.divider()
    st.header("📊 進行状況")
    if st.session_state.uploaded_files_list:
        total = len(st.session_state.uploaded_files_list)
        current = st.session_state.file_index + 1
        st.write(f"進捗: {current} / {total}")
        st.progress(current / total)
        
        # フォルダ内の移動ボタン
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ 前へ", disabled=(st.session_state.file_index == 0)):
                st.session_state.file_index -= 1
                st.rerun()
        with col_next:
            if st.button("次へ ➡️", disabled=(st.session_state.file_index == total - 1)):
                st.session_state.file_index += 1
                st.rerun()



# プロンプト定義
# 内容
# ひび割れ認識
PROMPT_FOR_NANOBANANA_V1 = """
写真にうつる建築物の表面を解析し，ひび割れを特定してください。
直線的なタイルやブロックの目地，建材の稜線，塗料の剥がれ部，異種材料の境界部分はひび割れではありません。
建材表面の幾何学的な模様や陰影はひび割れではありません。
特定したひび割れの上に、鮮明な赤色(透過率80%)の線を，ひび割れの太さに応じて描画した画像を生成してください。
元の画像と赤色の線のみで構成された画像を返してください。
"""
# 欠損・はがれ
PROMPT_FOR_NANOBANANA_V2 = """
写真にうつる建築物の表面を解析し，建材の欠損部や剥離部を特定してください。
欠損部や剥離部は寸法の縦横のアスペクト比0.5～2.0の範囲です．
アスペクト比が0.5未満の細長いものや、2.0を超える細長いものはひび割れの可能性が高いですが、今回は対象外としてください。
直線的なタイルやブロックの目地，建材の稜線，異種材料の境界部分は欠損部や剥離部ではありません。
建材表面の幾何学的な模様や陰影は欠損部や剥離部ではありません。
特定した欠損部や剥離部の上に、鮮明な赤色(透過率80%)を描画した画像を生成してください。
元の画像と赤色の描画領域のみで構成された画像を返してください。

"""
# 
PROMPT_FOR_NANOBANANA_V3 = """
写真にうつる建築物の表面を解析し，ひび割れを特定してください。
直線的なタイルやブロックの目地，建材の稜線，塗料の剥がれ部，異種材料の境界部分はひび割れではありません。
建材表面の幾何学的な模様や陰影はひび割れではありません。
特定したひび割れの上に、鮮明な赤色(透過率80%)の線を，ひび割れの太さに応じて描画した画像を生成してください。
元の画像と赤色の線のみで構成された画像を返してください。
"""
 

# プロンプト定義
PROMPT_MAP = {
    "ひび割れ": PROMPT_FOR_NANOBANANA_V1,
    "欠損・はがれ": PROMPT_FOR_NANOBANANA_V2,
    "その他": PROMPT_FOR_NANOBANANA_V3
}


# --- 1. ファイル一括読み込み ---
uploaded_files = st.file_uploader("解析する画像をまとめて選択してください", 
                                  type=["jpg", "png", "jpeg"], 
                                  accept_multiple_files=True)

# 新しいファイルがアップロードされたらリストを更新
if uploaded_files:
    if not st.session_state.uploaded_files_list or (len(uploaded_files) != len(st.session_state.uploaded_files_list)):
        st.session_state.uploaded_files_list = uploaded_files
        st.session_state.file_index = 0
        st.session_state.results_dict = {} # リセット

# --- 2. 個別処理エリア ---
if st.session_state.uploaded_files_list:
    current_file = st.session_state.uploaded_files_list[st.session_state.file_index]
    filename = current_file.name
    
    st.subheader(f"📂 処理中: {filename}")
    
    col_l, col_r = st.columns([1, 1])
    
    # 画像の読み込み
    pil_img = Image.open(current_file)
    w, h = pil_img.size
    
    with col_l:
        st.image(pil_img, caption="オリジナル画像", use_column_width=True)
        if st.button("🚀 この画像を 解析", use_container_width=True):
            if not api_key:
                st.error("APIキーを入力してください")
            else:
                with st.spinner("解析中..."):
                    current_file.seek(0)
                    traced_data = logic.get_gemini_traced_image(
                        api_key, current_file.read(), 
                        logic.PROMPT_MAP[prompt_type] if hasattr(logic, 'PROMPT_MAP') else "ひび割れを赤線で引いて", 
                        model_id
                    )
                    # 結果を一時保存
                    st.session_state.results_dict[filename] = {"traced_data": traced_data}
                    st.success("解析完了！右側で修正してください。")

    with col_r:
        if filename in st.session_state.results_dict and st.session_state.results_dict[filename].get("traced_data"):
            res = st.session_state.results_dict[filename]
            traced_pil = Image.open(io.BytesIO(res["traced_data"]))
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                background_image=traced_pil,
                update_streamlit=True,
                height=h * (600 / w) if w > 600 else h,
                width=600 if w > 600 else w,
                drawing_mode="polygon",
                key=f"canvas_{filename}", # ファイルごとにキーを変えるのがコツ
            )
            st.caption("誤検知エリアをマウスで囲んでください")

            # 描画済みマスク（アルファチャンネル）を取得するヘルパー
            def get_exclusion_mask(image_data, target_w, target_h):
                """canvas の image_data から除外マスクを target サイズで返す"""
                if image_data is None:
                    return None
                alpha = image_data[:, :, 3]
                if alpha.max() == 0:
                    return None
                return cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # 誤検知除外後プレビュー
            if canvas_result.image_data is not None:
                mask = get_exclusion_mask(canvas_result.image_data, w, h)
                if mask is not None:
                    traced_np = np.array(traced_pil.convert("RGB").resize((w, h), Image.LANCZOS))
                    orig_np = np.array(pil_img.convert("RGB"))
                    result_np = np.where(mask[:, :, np.newaxis] > 0, orig_np, traced_np)
                    st.image(Image.fromarray(result_np.astype(np.uint8)), caption="誤検知除外後プレビュー", use_column_width=True)

            if st.button("✅ この画像の結果を確定して保存", use_container_width=True):
                # 除外マスクを traced_data に適用してから YOLO 変換
                traced_bytes_to_use = res["traced_data"]
                if canvas_result.image_data is not None:
                    nparr = np.frombuffer(traced_bytes_to_use, np.uint8)
                    traced_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    th, tw = traced_cv.shape[:2]
                    mask = get_exclusion_mask(canvas_result.image_data, tw, th)
                    if mask is not None:
                        orig_resized = np.array(pil_img.convert("RGB").resize((tw, th), Image.LANCZOS))[:, :, ::-1]
                        traced_cv[mask > 0] = orig_resized[mask > 0]
                    _, enc = cv2.imencode(".jpg", traced_cv)
                    traced_bytes_to_use = enc.tobytes()

                # YOLO変換実行
                yolo_txt, vis_img = logic.process_yolo_segmentation(
                    traced_bytes_to_use, w, h, min_area, []
                )
                # 最終結果を保存
                st.session_state.results_dict[filename].update({
                    "yolo_txt": yolo_txt,
                    "vis_img": vis_img,
                    "completed": True
                })
                st.success(f"{filename} のラベルを保存しました！")

# --- 3. 一括書き出しエリア ---
if st.session_state.results_dict:
    st.divider()
    st.subheader("📦 一括書き出し")
    
    completed_count = sum(1 for r in st.session_state.results_dict.values() if r.get("completed"))
    st.write(f"確定済み画像: {completed_count} / {len(st.session_state.uploaded_files_list)}")

    if st.button("📁 全ての確定済みデータをZIPでダウンロード", use_container_width=True):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for fname, data in st.session_state.results_dict.items():
                if data.get("completed"):
                    # ラベル保存
                    zf.writestr(f"labels/{Path(fname).stem}.txt", data["yolo_txt"])
                    # 確認用画像保存
                    import cv2
                    _, img_enc = cv2.imencode(".jpg", data["vis_img"])
                    zf.writestr(f"visualized/{fname}", img_enc.tobytes())
                    # 元画像も同梱（任意）
                    # zf.writestr(f"images/{fname}", ...) 

        st.download_button(
            label="🔥 ZIPをダウンロード",
            data=zip_buffer.getvalue(),
            file_name="yolo_dataset_all.zip",
            mime="application/zip",
            use_container_width=True
        )
