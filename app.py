import gradio as gr
from piper import PiperVoice, SynthesisConfig
from pathlib import Path
import numpy as np
import requests
import threading
import re

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

_voice_registry = {}
_download_tasks = {}  # {model_name: {"progress": 0..1, "msg": str, "error": str|None}}

# CJK 字符範圍：中文/日文/韓文
_CJK_CHARS = (
    "一-鿿㐀-䶿"    # 中文
    "぀-ゟ゠-ヿ"    # 日文
    "가-힯ᄀ-ᇿ"    # 韓文
    "　-〿"        # CJK 標點符號 (。、「」《》等)
    "！-～"        # 全形標點 (，！？：；（）等)
)
_CJK_TOKEN = re.compile(f"[{_CJK_CHARS}]+")
_SPLIT_RE = re.compile(f"([{_CJK_CHARS}]+|[^{_CJK_CHARS}]+)")


def _load_registry():
    global _voice_registry
    if _voice_registry:
        return _voice_registry
    try:
        r = requests.get(VOICES_JSON_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        for key, info in data.items():
            path = list(info["files"].keys())[0]
            _voice_registry[key] = {
                "path": path,
                "size": info["files"][path]["size_bytes"],
                "lang": info["language"]["name_english"],
                "lang_code": info["language"]["code"],
                "quality": info["quality"],
            }
    except Exception:
        pass
    return _voice_registry


def _installed_set():
    return {f.stem for f in MODELS_DIR.glob("*.onnx")}


def _installed_choices():
    return sorted(_installed_set())


def _cjk_model_choices():
    """已安裝的 CJK 模型（中文/日文/韓文）"""
    return sorted(n for n in _installed_set() if n.startswith(("zh_", "ja_", "ko_")))


def _latin_model_choices():
    """已安裝的非 CJK 模型（英文及歐洲語言）"""
    return sorted(n for n in _installed_set() if not n.startswith(("zh_", "ja_", "ko_")))


def _build_available_table(search="", lang_filter=""):
    registry = _load_registry()
    installed = _installed_set()
    rows = []
    for name, info in sorted(registry.items()):
        if search and search.lower() not in name.lower():
            continue
        if lang_filter and not info["lang_code"].startswith(lang_filter):
            continue
        rows.append([
            name,
            info["lang"],
            info["quality"],
            f"{info['size'] / 1024 / 1024:.1f} MB",
            "✓ 已安裝" if name in installed else "—",
        ])
    return rows


def _build_installed_table():
    registry = _load_registry()
    rows = []
    for f in sorted(MODELS_DIR.glob("*.onnx"), key=lambda x: x.stat().st_size, reverse=True):
        name = f.stem
        info = registry.get(name, {})
        rows.append([
            name,
            info.get("lang", "?"),
            info.get("quality", "?"),
            f"{f.stat().st_size / 1024 / 1024:.1f} MB",
        ])
    return rows


def _build_download_progress_table():
    rows = []
    for name, task in list(_download_tasks.items()):
        pct = f"{task['progress'] * 100:.0f}%" if task["progress"] > 0 else "0%"
        rows.append([name, pct, task["msg"]])
    return rows


def _refresh_all():
    return (
        _installed_choices(),
        _cjk_model_choices(),
        _latin_model_choices(),
        _build_installed_table(),
        _build_available_table(),
        _build_download_progress_table(),
    )


# === 混合語言支援 ===

def _split_mixed_text(text):
    """正則直接按 CJK/非CJK 邊界切分，中文優先，不合併不同腳本。

    例如: "go to school 等等" → [("latin", "go to school "), ("cjk", "等等")]
    而不是把 "等等" 錯誤合併進英文段。
    """
    if not text:
        return []
    segments = []
    for token in _SPLIT_RE.findall(text):
        token = token.strip()
        if not token:
            continue
        if _CJK_TOKEN.search(token):
            segments.append(("cjk", token))
        else:
            segments.append(("latin", token))
    return segments


def _synthesize_segments(segments, cjk_model, en_model, syn_config, progress=None):
    """Synthesize mixed segments and return combined audio + status lines."""
    results = []
    lines = []
    sample_rate = 22050
    total = len(segments)

    for i, (st, seg) in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue

        if st == "cjk":
            model = cjk_model
            tag = "CJK"
        else:
            model = en_model
            tag = "Latin"

        if progress is not None:
            progress((i + 1) / total, desc=f"混合合成 [{tag}] {i+1}/{total} - {model}")

        if model is None:
            lines.append(f"⚠ 跳過（缺{tag}模型）: {seg[:40]}...")
            continue

        onnx_path = MODELS_DIR / f"{model}.onnx"
        if not onnx_path.exists():
            lines.append(f"⚠ 跳過（找不到{model}）: {seg[:40]}...")
            continue

        try:
            voice = PiperVoice.load(str(onnx_path))
            chunks = list(voice.synthesize(seg, syn_config))
            for ch in chunks:
                sr = ch.sample_rate
                if ch._audio_int16_array is not None:
                    results.append(ch._audio_int16_array)
                elif ch.audio_float_array is not None:
                    results.append((ch.audio_float_array * 32767).astype(np.int16))
                sample_rate = sr
            lines.append(f"  [{tag}] {model}: {seg[:50]}{'...' if len(seg) > 50 else ''}")
        except Exception as e:
            lines.append(f"✗ [{tag}] 合成失敗: {e}")

    if not results:
        return None, sample_rate, "\n".join(lines)

    # 30ms minimal gap — inaudible, just prevents audio clicks at segment boundaries
    silence = np.zeros(int(sample_rate * 0.03), dtype=np.int16)
    combined = results[0]
    for arr in results[1:]:
        combined = np.concatenate([combined, silence, arr])

    return combined, sample_rate, "\n".join(lines)


# === event handlers ===

def download_model(model_name):
    name = (model_name or "").strip()
    if not name:
        return (*_refresh_all(), "請輸入模型名稱")
    registry = _load_registry()
    if name not in registry:
        return (*_refresh_all(), f"找不到模型: {name}")
    if name in _installed_set():
        return (*_refresh_all(), f"{name} 已安裝，無需重複下載")
    if name in _download_tasks and _download_tasks[name]["progress"] < 1.0:
        return (*_refresh_all(), f"{name} 正在下載中…")

    info = registry[name]
    _download_tasks[name] = {"progress": 0.0, "msg": "排隊中…", "error": None}

    def _do_download():
        try:
            onnx_path = MODELS_DIR / f"{name}.onnx"
            json_path = MODELS_DIR / f"{name}.onnx.json"
            if not onnx_path.exists():
                url = f"{PIPER_VOICES_BASE}/{info['path']}"
                r = requests.get(url, stream=True, timeout=300)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                tmp_path = onnx_path.with_suffix(".onnx.tmp")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            _download_tasks[name]["progress"] = downloaded / total
                            _download_tasks[name]["msg"] = (
                                f"下載中 {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
                            )
                tmp_path.rename(onnx_path)
            if not json_path.exists():
                config_url = f"{PIPER_VOICES_BASE}/{info['path'].replace('.onnx', '.onnx.json')}"
                r = requests.get(config_url, timeout=60)
                r.raise_for_status()
                json_path.write_bytes(r.content)
            _download_tasks[name]["progress"] = 1.0
            _download_tasks[name]["msg"] = f"✓ 完成 ({info['size'] / 1024 / 1024:.1f} MB)"
            _download_tasks[name]["error"] = None
        except Exception as e:
            tmp = MODELS_DIR / f"{name}.onnx.tmp"
            tmp.unlink(missing_ok=True)
            _download_tasks[name]["progress"] = 1.0
            _download_tasks[name]["msg"] = f"✗ 失敗: {e}"
            _download_tasks[name]["error"] = str(e)

    threading.Thread(target=_do_download, daemon=True).start()
    # Download status only goes to download_progress_table, NOT status_text
    return (*_refresh_all(), "")


def delete_model(model_name):
    if not model_name:
        return (*_refresh_all(), "請輸入要刪除的模型名稱")
    name = model_name.strip()
    onnx_path = MODELS_DIR / f"{name}.onnx"
    json_path = MODELS_DIR / f"{name}.onnx.json"
    try:
        onnx_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return (*_refresh_all(), f"✓ {name} 已刪除")
    except Exception as e:
        return (*_refresh_all(), f"✗ 刪除失敗: {e}")


def refresh_available(search, lang_filter):
    return _build_available_table(search, lang_filter)


def synthesize(text, model_name, speed, noise_scale, noise_w, volume,
               mixed_mode, cjk_model, en_model, progress=gr.Progress()):
    """Normal or mixed-language synthesis."""
    if not text or not text.strip():
        return None, "請輸入文本"
    if not model_name:
        return None, "請選擇或輸入一個模型名稱"

    config = SynthesisConfig(
        length_scale=1.0 / speed if speed > 0 else 1.0,
        noise_scale=noise_scale,
        noise_w_scale=noise_w,
        volume=volume,
    )

    # ---- 混合語言模式 ----
    if mixed_mode:
        segments = _split_mixed_text(text)
        has_cjk = any(st == "cjk" for st, _ in segments)
        has_latin = any(st == "latin" for st, _ in segments)

        if not has_cjk or not has_latin:
            return None, "文本未檢測到中英混合。請關閉混合模式使用普通合成，或檢查文本內容。"

        if not cjk_model:
            return None, "混合模式需要指定中文模型。請從下方下拉框選擇。"
        if not en_model:
            return None, "混合模式需要指定英文模型。請從下方下拉框選擇。"

        progress(0, desc=f"混合合成 - 共 {len(segments)} 段")
        audio, sr, detail = _synthesize_segments(segments, cjk_model, en_model, config, progress)
        if audio is None:
            return None, detail
        return (sr, audio), f"✓ 混合合成完成 ({len(audio) / sr:.1f}s)\n{detail}"

    # ---- 普通模式 ----
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    if not onnx_path.exists():
        return None, f"找不到模型文件: {onnx_path}"

    try:
        progress(0.3, desc="載入模型中...")
        voice = PiperVoice.load(str(onnx_path))
        progress(0.6, desc="合成語音中...")
        chunks = list(voice.synthesize(text, config))
        if not chunks:
            return None, "未能生成音頻"

        sr = chunks[0].sample_rate
        all_audio = []
        for ch in chunks:
            if ch._audio_int16_array is not None:
                all_audio.append(ch._audio_int16_array)
            elif ch.audio_float_array is not None:
                all_audio.append((ch.audio_float_array * 32767).astype(np.int16))

        combined = np.concatenate(all_audio)
        progress(1.0, desc="完成")
        return (sr, combined), f"✓ 生成完成 ({len(combined) / sr:.1f}s)"
    except Exception as e:
        import traceback
        return None, f"✗ 生成失敗: {traceback.format_exc()}"


def load_txt(file_bytes):
    if file_bytes is None:
        return ""
    return file_bytes.decode("utf-8")


# === UI ===

with gr.Blocks(title="Piper TTS — 語音合成") as demo:
    gr.Markdown("# Piper TTS — 語音合成")

    # ============================================
    # 上部：輸入 + 輸出
    # ============================================
    with gr.Row():
        with gr.Column(scale=2):
            # 模型選擇
            with gr.Row():
                installed_dropdown = gr.Dropdown(
                    choices=_installed_choices(),
                    label="選擇已安裝模型",
                    interactive=True,
                    allow_custom_value=True,
                    scale=3,
                )
            gr.Markdown("*下拉選擇或直接輸入模型名稱（如 `zh_CN-huayan-medium`）*")

            # 混合語言模式
            mixed_checkbox = gr.Checkbox(
                value=False,
                label="啟用混合語言模式（中英混輸時自動分段，用不同模型合成後拼接）",
            )
            with gr.Row():
                cjk_model_dd = gr.Dropdown(
                    choices=_cjk_model_choices(),
                    label="中文段落使用模型",
                    interactive=True,
                    allow_custom_value=True,
                    scale=1,
                )
                en_model_dd = gr.Dropdown(
                    choices=_latin_model_choices(),
                    label="英文段落使用模型",
                    interactive=True,
                    allow_custom_value=True,
                    scale=1,
                )

            text_input = gr.TextArea(
                label="輸入文本",
                placeholder="在此輸入或粘貼要合成語音的文本…",
                lines=6,
            )
            txt_upload = gr.File(
                label="上傳 TXT 文件（內容自動填充到文本框）",
                file_types=[".txt"],
                type="binary",
            )

            with gr.Accordion("語音參數", open=False):
                with gr.Row():
                    speed_slider = gr.Slider(0.5, 3.0, 1.0, step=0.1, label="語速")
                    volume_slider = gr.Slider(0.1, 2.0, 1.0, step=0.1, label="音量")
                with gr.Row():
                    noise_slider = gr.Slider(0.1, 1.5, 0.667, step=0.01, label="韻律變化")
                    noise_w_slider = gr.Slider(0.1, 1.5, 0.8, step=0.01, label="音素變化")

            # 生成按鈕 + 生成狀態（獨立的狀態欄）
            with gr.Row():
                generate_btn = gr.Button("生成語音", variant="primary", size="lg")
                status_text = gr.Textbox(label="生成狀態", interactive=False, scale=3)

        with gr.Column(scale=1):
            gr.Markdown("### 輸出")
            audio_output = gr.Audio(label="音頻", type="numpy")

    # ============================================
    # 下部：模型庫
    # ============================================
    gr.Markdown("---")

    # ---- 模型庫（左） + 下載進度（右）獨立並排 ----
    with gr.Row():
        # 左側：模型庫
        with gr.Column(scale=3):
            gr.Markdown("## 模型庫（來自 HuggingFace）")
            with gr.Row():
                model_search = gr.Textbox(
                    label="搜尋模型",
                    placeholder="輸入關鍵字篩選，例如: en_US, zh_CN, lessac…",
                    scale=3,
                )
                lang_filter_dd = gr.Dropdown(
                    choices=["", "en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "nl", "ru", "ar"],
                    value="",
                    label="語言",
                    scale=1,
                )

            available_table = gr.Dataframe(
                headers=["模型名稱", "語言", "品質", "大小", "狀態"],
                value=_build_available_table(),
                label=f"全部可用模型（共 {len(_voice_registry)} 個）",
                interactive=False,
                row_count=(10, "dynamic"),
                wrap=True,
            )

            with gr.Row():
                with gr.Column(scale=2):
                    download_name = gr.Textbox(
                        label="輸入要下載的模型名稱",
                        placeholder="從上表複製模型名稱到此處…",
                    )
                with gr.Column(scale=1):
                    download_btn = gr.Button("下載模型", variant="secondary")

        # 右側：下載進度（獨立 block）
        with gr.Column(scale=1):
            gr.Markdown("## 下載進度")
            download_progress_table = gr.Dataframe(
                headers=["模型名稱", "進度", "狀態"],
                value=[],
                label="下載任務",
                interactive=False,
                row_count=(5, "dynamic"),
            )
            refresh_progress_btn = gr.Button("刷新進度", variant="secondary", size="sm")
            download_status = gr.Textbox(label="下載訊息", interactive=False)

    # ---- 已安裝 + 刪除 ----
    gr.Markdown("### 已安裝模型詳情")
    installed_table = gr.Dataframe(
        headers=["模型名稱", "語言", "品質", "大小"],
        value=_build_installed_table(),
        label="已安裝模型",
        interactive=False,
        row_count=(5, "dynamic"),
    )

    with gr.Row():
        with gr.Column(scale=2):
            delete_name = gr.Textbox(
                label="輸入要刪除的模型名稱",
                placeholder="從上表複製模型名稱到此處…",
            )
        with gr.Column(scale=1):
            delete_btn = gr.Button("刪除模型", variant="stop")

    # ============================================
    # events
    # ============================================

    demo.load(
        _refresh_all,
        outputs=[installed_dropdown, cjk_model_dd, en_model_dd, installed_table, available_table, download_progress_table],
    )

    model_search.change(
        refresh_available,
        inputs=[model_search, lang_filter_dd],
        outputs=[available_table],
    )
    lang_filter_dd.change(
        refresh_available,
        inputs=[model_search, lang_filter_dd],
        outputs=[available_table],
    )

    # 下載 → 只更新下載相關組件 + 模型列表，不碰生成狀態
    download_btn.click(
        download_model,
        inputs=[download_name],
        outputs=[installed_dropdown, cjk_model_dd, en_model_dd, installed_table, available_table, download_progress_table, download_status],
    )

    refresh_progress_btn.click(
        _refresh_all,
        outputs=[installed_dropdown, cjk_model_dd, en_model_dd, installed_table, available_table, download_progress_table],
    )

    # 刪除 → 只更新模型相關組件，不碰生成狀態
    delete_btn.click(
        delete_model,
        inputs=[delete_name],
        outputs=[installed_dropdown, cjk_model_dd, en_model_dd, installed_table, available_table, download_progress_table, download_status],
    )

    txt_upload.change(load_txt, inputs=[txt_upload], outputs=[text_input])

    # 生成 → 只更新音頻 + 生成狀態，不碰下載區
    generate_btn.click(
        synthesize,
        inputs=[
            text_input, installed_dropdown,
            speed_slider, noise_slider, noise_w_slider, volume_slider,
            mixed_checkbox, cjk_model_dd, en_model_dd,
        ],
        outputs=[audio_output, status_text],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
