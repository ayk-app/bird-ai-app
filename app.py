import os
import sys
import subprocess
import tempfile

# 他人のPCでも目の前のパーツを強制認識させるセーフティネット
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

SITE_PACKAGES = os.path.join(BASE_DIR, "my_env", "Lib", "site-packages")
if os.path.exists(SITE_PACKAGES):
    sys.path.insert(0, SITE_PACKAGES)
import streamlit as st
import os
import shutil
import urllib.request

# --- AIモデル & 辞書ファイル自動セットアップ ---
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. ルートにある辞書(ja.txt)を model フォルダへ自動複製
for file_name in ["ja.txt", "labels.txt"]:
    if os.path.exists(file_name):
        shutil.copy(file_name, f"{MODEL_DIR}/{file_name}")

# 2. AIモデル(tflite)の自動ダウンロード
MODEL_PATH = f"{MODEL_DIR}/audio-model.tflite"
MODEL_URL = "https://github.com/ayk-app/bird-ai-app/releases/download/v1.0/audio-model.tflite"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
# ----------------------------------------------
st.set_page_config(page_title="北海道の野鳥AI", page_icon="🦉", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F4F9F4; }
    h1, h2, h3, h4 { color: #4A3525 !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #2A402B; }
    [data-testid="stSidebar"] * { color: #E8F0E8 !important; }
    [data-testid="stTable"] { border: 2px solid #6E533C; border-radius: 10px; background-color: #FFFFFF; }
    hr { border-bottom: 2px dashed #8F7256 !important; }
    .env-tag {
        display: inline-block;
        background-color: #6E533C;
        color: #F4F9F4;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import scipy.signal as signal
import pandas as pd
import scipy.io.wavfile as wavf

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

st.title("🌲 北海道の野鳥・完全AI識別ツール 🦉")
st.write("タイムラインにはAIの微かな予想もすべて表示し、上の表には設定値以上の鳥だけを厳格に集計します。")
st.markdown("**【AIの自信度】** 🔴ほぼ確定(50%〜) / 🟠かなり有力(20%〜) / 🟢可能性あり(10%〜) / ⚪参考(10%未満)")

MODEL_NAME = os.path.join("model", "audio-model.tflite")
LABEL_FILE = os.path.join("model", "ja.txt")

if not os.path.exists(MODEL_NAME) or not os.path.exists(LABEL_FILE):
    st.error("❌ AIモデルまたは日本語辞書が見つかりません。『model』フォルダの中に入っているか確認してください。")
    st.stop()

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f.readlines()]

def load_bird_list(candidates, list_name):
    birds = set()
    found_file = None
    for f in candidates:
        if os.path.exists(f):
            found_file = f
            break
    
    if found_file:
        try:
            if found_file.endswith('.csv'):
                try:
                    df = pd.read_csv(found_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(found_file, encoding='shift_jis')
                
                if '鳥類和名' in df.columns:
                    col = '鳥類和名'
                elif '和名' in df.columns:
                    col = '和名'
                elif '名前' in df.columns:
                    col = '名前'
                else:
                    col = df.columns[0]
                    
                birds = set([str(x).strip() for x in df[col].dropna()])
            else:
                with open(found_file, 'r', encoding='utf-8') as f:
                    birds = set([line.strip() for line in f.readlines() if line.strip()])
            st.sidebar.success(f"✅ 『{os.path.basename(found_file)}』を認識 ({len(birds)}種)")
            return birds, True
        except Exception as e:
            st.sidebar.error(f"⚠️ {list_name}の読み込み失敗: {e}")
            return set(), False
    else:
        st.sidebar.error(f"⚠️ {list_name}が見つかりません。『data』フォルダ内を確認してください。")
        return set(), False

st.sidebar.header("🪓 解析設定")

st.sidebar.subheader("📍 録音した環境")
record_env = st.sidebar.selectbox(
    "どこで録音したデータですか？",
    [
        "指定しない（すべて表示）",
        "🏡 庭・市街地・公園",
        "🌲 深い森・山の中・笹ヤブ",
        "🌊 水辺・海岸・湖畔",
        "🌾 草地・ひらけた農耕地"
    ],
    help="ここで選んだ環境がレポートに記録され、似た鳥の判別ヒントになります。"
)
st.sidebar.markdown("---")

threshold = st.sidebar.slider("総合結果に載せるAIの感度 (％)", min_value=5, max_value=100, value=15, step=1)
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 抽出フィルター（レッドリスト等）")

use_whitelist = st.sidebar.checkbox("北海道の鳥だけに絞り込む", value=True)
whitelist_birds = set()
if use_whitelist:
    whitelist_birds, success = load_bird_list([
        os.path.join("data", "北海道の鳥類.csv"), os.path.join("data", "hokkaido_birds.txt"),
        "北海道の鳥類.csv", "hokkaido_birds.txt"
    ], "北海道の鳥類")
    if not success: use_whitelist = False

use_hkd_redlist = st.sidebar.checkbox("🔴 北海道レッドリスト種のみ抽出", value=False)
hkd_red_birds = set()
if use_hkd_redlist:
    hkd_red_birds, success = load_bird_list([
        os.path.join("data", "北海道レッドリスト.csv"), os.path.join("data", "hkd_redlist.txt"),
        "北海道レッドリスト.csv", "hkd_redlist.txt"
    ], "北海道レッドリスト")
    if not success: use_hkd_redlist = False

use_env_redlist = st.sidebar.checkbox("🔴 環境省レッドリスト種のみ抽出", value=False)
env_red_birds = set()
if use_env_redlist:
    env_red_birds, success = load_bird_list([
        os.path.join("data", "環境省レッドリスト.csv"), os.path.join("data", "env_redlist.txt"),
        "環境省レッドリスト.csv", "env_redlist.txt"
    ], "環境省レッドリスト")
    if not success: use_env_redlist = False

st.sidebar.markdown("---")
st.sidebar.subheader("🧬 オリジナル特殊検知エンジン")
use_custom_engine = st.sidebar.checkbox(
    "特殊エンジンを【補助モード】で使う", 
    value=False, 
    help="メインAIが設定値未満で『該当なし』の時だけ、見本と照らし合わせてヒントを出します。"
)
match_threshold = st.sidebar.slider("オリジナル見本との一致度 (％)", min_value=40, max_value=100, value=75, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔇 ノイズ除去設定")
use_noise_reduction = st.sidebar.checkbox("低音ノイズを除去する (風・車など)", value=True)
cutoff_freq = st.sidebar.slider("カットする周波数 (Hz)", min_value=100, max_value=2000, value=500, step=50)

SAMPLE_DIR = "custom_samples"
@st.cache_data
def load_custom_templates():
    templates = {}
    total_files = 0
    if os.path.exists(SAMPLE_DIR):
        for species_folder in os.listdir(SAMPLE_DIR):
            species_path = os.path.join(SAMPLE_DIR, species_folder)
            if os.path.isdir(species_path):
                templates[species_folder] = []
                for file_name in os.listdir(species_path):
                    if file_name.lower().endswith(('.mp3', '.wav', '.m4a')):
                        file_path = os.path.join(species_path, file_name)
                        try:
                            y_sub, _ = librosa.load(file_path, sr=48000, duration=3.0)
                            if len(y_sub) < 144000:
                                padded = np.zeros(144000, dtype=np.float32)
                                padded[:len(y_sub)] = y_sub
                                y_sub = padded
                            else:
                                y_sub = y_sub[:144000]
                            S = librosa.feature.melspectrogram(y=y_sub, sr=48000, n_mels=128)
                            S_dB = librosa.power_to_db(S, ref=np.max)
                            templates[species_folder].append(S_dB)
                            total_files += 1
                        except:
                            pass
                if not templates[species_folder]:
                    del templates[species_folder]
    return templates, total_files

custom_templates, total_custom_files = load_custom_templates()
if custom_templates:
    loaded_species = " / ".join(custom_templates.keys())
    st.sidebar.info(f"✨ オリジナル見本を記憶: {len(custom_templates)}種 ({total_custom_files}ファイル)\n\n【内訳】\n{loaded_species}")

def format_time(seconds_float):
    m = int(seconds_float // 60)
    s = seconds_float % 60
    if m > 0:
        return f"{m}分{s:04.1f}秒"
    else:
        return f"{s:.1f}秒"

# ─── メイン処理 ───
uploaded_file = st.file_uploader("📂 音声・動画ファイルを選択してください (最大500MB)", type=["wav", "mp3", "m4a", "mp4", "mov"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
with st.spinner("データを解読・ノイズ処理中..."):
            try:
                ext = uploaded_file.name.split('.')[-1].lower()
                if ext in ['wav']:
                    audio_bytes = io.BytesIO(uploaded_file.getbuffer())
                    y, sr = librosa.load(audio_bytes, sr=48000)
                else:
                    if shutil.which("ffmpeg"):
                        ffmpeg_path = "ffmpeg"
                    else:
                        ffmpeg_path = os.path.join(BASE_DIR, "tools", "ffmpeg.exe")
                        if not os.path.exists(ffmpeg_path):
                            ffmpeg_path = os.path.join(BASE_DIR, "ffmpeg.exe")

                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
                            tmp_in.write(uploaded_file.getbuffer())
                            tmp_in_path = tmp_in.name
                        tmp_out_path = tmp_in_path + ".wav"
                        subprocess.run([ffmpeg_path, "-y", "-i", tmp_in_path, "-ar", "48000", tmp_out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        y, sr = librosa.load(tmp_out_path, sr=48000)
            except Exception as e:
                st.error(f"読み込みエラー: {e}")
                st.stop()

            if use_noise_reduction:
                nyq = 0.5 * sr
                normal_cutoff = cutoff_freq / nyq
                b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
                y = signal.filtfilt(b, a, y).astype(np.float32)

            total_samples = len(y)
            duration = total_samples / sr
            st.info(f"🎵 解析対象の長さ: {format_time(duration)}")
            
        except Exception as e:
            st.error(f"⚠️ 読み込みエラー: {e}")
            st.stop()
            
    with st.spinner("🦅 内蔵AI＆特殊波形照合エンジンがスキャン中..."):
        try:
            interpreter = tflite.Interpreter(model_path=MODEL_NAME)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            window_samples = 144000
            num_windows = int(np.ceil(total_samples / window_samples))
            
            detected_counts = {}  
            timeline_data = []
            chart_data = []
            best_clips = {}
            
            for i in range(num_windows):
                start_idx = i * window_samples
                end_idx = min(start_idx + window_samples, total_samples)
                
                chunk = np.zeros(window_samples, dtype=np.float32)
                chunk[:end_idx - start_idx] = y[start_idx:end_idx]
                
                dummy_input = np.expand_dims(chunk, axis=0)
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                raw_outputs = interpreter.get_tensor(output_details[0]['index'])[0]
                
                all_predictions = []
                for idx, raw_score in enumerate(raw_outputs):
                    confidence = (1 / (1 + np.exp(-raw_score))) * 100
                    raw_label = labels[idx] if idx < len(labels) else f"Unknown ID: {idx}"
                    jp_name = raw_label.split("_")[-1] if "_" in raw_label else raw_label
                    
                    if use_whitelist and (jp_name not in whitelist_birds): continue
                    if use_hkd_redlist and (jp_name not in hkd_red_birds): continue
                    if use_env_redlist and (jp_name not in env_red_birds): continue
                    
                    if confidence >= 1.0:
                        all_predictions.append((raw_label, confidence))
                
                all_predictions.sort(key=lambda x: x[1], reverse=True)
                
                top_main_score = all_predictions[0][1] if all_predictions else 0
                
                max_sim = 0
                best_match_species = None
                
                if use_custom_engine and custom_templates and (top_main_score < threshold):
                    current_mel = librosa.feature.melspectrogram(y=chunk, sr=48000, n_mels=128)
                    current_mel_dB = librosa.power_to_db(current_mel, ref=np.max)
                    
                    for species_name, templates in custom_templates.items():
                        for template in templates:
                            corr = np.corrcoef(current_mel_dB.flatten(), template.flatten())[0, 1]
                            sim_percent = max(0, corr * 100)
                            if sim_percent > max_sim:
                                max_sim = sim_percent
                                best_match_species = species_name

                custom_passed = True
                if best_match_species:
                    if use_whitelist and (best_match_species not in whitelist_birds): custom_passed = False
                    if use_hkd_redlist and (best_match_species not in hkd_red_birds): custom_passed = False
                    if use_env_redlist and (best_match_species not in env_red_birds): custom_passed = False

                timeline_texts = []

                for bird_label, confidence in all_predictions:
                    if confidence >= threshold:
                        jp_name = bird_label.split("_")[-1] if "_" in bird_label else bird_label
                        detected_counts[bird_label] = detected_counts.get(bird_label, 0) + 1
                        
                        chart_data.append({
                            "時間 (秒)": start_idx / sr,
                            "鳥の種類": jp_name,
                            "確信度 (%)": confidence
                        })
                        if jp_name not in best_clips or confidence > best_clips[jp_name][0]:
                            best_clips[jp_name] = (confidence, chunk.copy())
                        
                if best_match_species and max_sim >= match_threshold and custom_passed:
                    custom_label = f"💡 [特殊ヒント] {best_match_species}"
                    detected_counts[custom_label] = detected_counts.get(custom_label, 0) + 1
                    
                    chart_data.append({
                        "時間 (秒)": start_idx / sr,
                        "鳥の種類": custom_label,
                        "確信度 (%)": max_sim
                    })
                    if custom_label not in best_clips or max_sim > best_clips[custom_label][0]:
                        best_clips[custom_label] = (max_sim, chunk.copy())

                if all_predictions:
                    top_3 = all_predictions[:3]
                    formatted_preds = []
                    for b, c in top_3:
                        name = b.split('_')[-1]
                        if c >= 50.0: icon = "🔴"
                        elif c >= 20.0: icon = "🟠"
                        elif c >= 10.0: icon = "🟢"
                        else: icon = "⚪"
                        formatted_preds.append(f"{icon}{name} ({c:.1f}%)")
                        
                    ai_str = " / ".join(formatted_preds)
                    timeline_texts.append(f"🤖 AI予想: {ai_str}")
                
                if best_match_species and max_sim >= 40 and custom_passed:
                    timeline_texts.append(f"💡 特殊波形ヒント: {best_match_species} ({max_sim:.1f}%)")

                birds_str = " \n".join(timeline_texts) if timeline_texts else "（該当する野鳥なし）"
                start_time = start_idx / sr
                end_time = end_idx / sr
                time_range_str = f"{format_time(start_time)} ～ {format_time(end_time)}"
                
                timeline_data.append({"時間帯": time_range_str, "AIの解析メモ": birds_str})
            
            # 1. 総合解析結果
            with st.expander("📋 総合解析結果（正式集計）", expanded=True):
                if record_env != "指定しない（すべて表示）":
                    st.markdown(f'<div class="env-tag">📍 記録環境：{record_env}</div>', unsafe_allow_html=True)
                    
                if detected_counts:
                    summary_data = []
                    for bird_label, count in detected_counts.items():
                        time_percentage = (count / num_windows) * 100
                        summary_data.append({
                            "鳥の種類": bird_label,
                            "出現時間割合": f"{time_percentage:.1f} %",
                            "検出コマ数": f"{count} / {num_windows} 回",
                            "_sort_key": time_percentage
                        })
                    summary_data.sort(key=lambda x: x["_sort_key"], reverse=True)
                    for d in summary_data: d.pop("_sort_key")
                    st.dataframe(summary_data, use_container_width=True)
                else:
                    st.warning(f"⚠️ 指定したフィルター条件に該当する野鳥は見つかりませんでした。")
            
            # 2. 音声切り抜き
            with st.expander("🎧 証拠音声の自動切り抜き（ハイライト再生）", expanded=True):
                st.write("AIが「最も自信を持って判定した瞬間（ベストショット）」の音声を、鳥ごとに再生できます。")
                if best_clips:
                    sorted_clips = sorted(best_clips.items(), key=lambda x: x[1][0], reverse=True)
                    cols = st.columns(3)
                    for idx, (b_label, (conf, audio_chunk)) in enumerate(sorted_clips):
                        jp_name = b_label.split('_')[-1] if '_' in b_label else b_label
                        with cols[idx % 3]:
                            st.markdown(f"**{jp_name}** ({conf:.1f}%)")
                            audio_int16 = np.clip(audio_chunk * 32767, -32768, 32767).astype(np.int16)
                            buf = io.BytesIO()
                            wavf.write(buf, sr, audio_int16)
                            st.audio(buf, format="audio/wav")
                else:
                    st.info("切り抜く音声がありませんでした。")

            # 3. グラフ
            with st.expander("📈 鳥たちの合唱タイムライン（グラフ）", expanded=False):
                st.write("どの時間帯に、どの鳥が鳴いていたか一目で分かります。点（●）の大きさがAIの確信度を表します。")
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    st.scatter_chart(chart_df, x="時間 (秒)", y="鳥の種類", size="確信度 (%)", color="鳥の種類", height=400)
                else:
                    st.info("グラフ化するデータがありません。")

            # 4. タイムライン
            with st.expander("⏱️ 野鳥の出現タイムライン", expanded=False):
                st.dataframe(timeline_data, use_container_width=True, height=500)
            
        except Exception as e:
            st.error(f"⚠️ 計算エラー: {e}")

    # 5. スペクトログラム
    with st.expander("📊 ノイズ処理後の音声スペクトログラム", expanded=False):
        with st.spinner("グラフを描画中..."):
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
