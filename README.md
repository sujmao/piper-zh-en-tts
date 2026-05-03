# Piper TTS — 中英混合語音合成 Web 應用

基於 [Piper TTS](https://github.com/rhasspy/piper) 的 Gradio Web 圖形界面，無需命令列操作，打開瀏覽器即可將文字轉換為自然語音。支援中英混合文本自動分段與雙語模型合成。

## 快速開始

```bash
# 1. 安裝依賴（僅首次）
pip install piper-tts gradio numpy requests

# 2. 啟動服務
python app.py

# 3. 打開瀏覽器訪問
# http://127.0.0.1:7860
```

## 功能

- **語音合成**：輸入文字，選擇模型，生成自然語音（輸出 WAV 格式）
- **混合語言模式**：中英混輸時自動分段，分別用不同語音模型合成後拼接
- **模型管理**：瀏覽 HuggingFace 模型庫、一鍵下載、查看與刪除已安裝模型
- **文本上傳**：支援直接上傳 .txt 文件
- **參數調整**：語速、音量、韻律變化、音素變化

## 混合語言模式

當文本同時包含中文（或日文/韓文）和英文時，可使用混合語言模式：

1. 勾選「啟用混合語言模式」
2. 選擇 CJK 模型（如 `zh_CN-huayan-medium`）和 Latin 模型（如 `en_US-lessac-medium`）
3. 輸入混合文本，點擊「生成語音」

分段規則：漢字/假名/諺文/CJK 標點 → CJK 模型；英文/數字/ASCII 標點 → Latin 模型。

## 推薦模型

| 語言 | 模型 | 大小 | 說明 |
|------|------|------|------|
| 中文 | `zh_CN-huayan-x_low` | 19.7 MB | 體積最小，適合快速試用 |
| 中文 | `zh_CN-huayan-medium` | 60.3 MB | 品質較好 |
| 中文 | `zh_CN-xiao_ya-medium` | 60.3 MB | 女聲 |
| 英文 | `en_US-lessac-medium` | 60.3 MB | 女聲 |
| 英文 | `en_US-ryan-medium` | 60.3 MB | 男聲 |

## 語音參數

| 參數 | 預設 | 說明 |
|------|------|------|
| 語速 | 1.0 | 0.5 = 半速，2.0 = 雙倍速 |
| 音量 | 1.0 | 0.5 = 減半，1.5 = 增強 |
| 韻律變化 | 0.667 | 控制語調高低起伏 |
| 音素變化 | 0.8 | 一般不需調整 |

新手建議保持預設值即可。

## 遷移到其他電腦

無硬編碼路徑，所有模型存放在程式目錄下的 `models/` 資料夾中：

1. 將整個程式資料夾複製到新電腦
2. 安裝 Python（建議 3.10 以上）
3. `pip install piper-tts gradio numpy requests`
4. `python app.py`

如果新電腦無法訪問 HuggingFace，可將 `models/` 資料夾一併複製，無需重新下載。支援 Windows / macOS / Linux。

## 常見問題

**Q: 生成失敗，提示「找不到模型」？**  
A: 請先在下方模型庫中下載模型。

**Q: 下載模型失敗？**  
A: 模型文件託管在 HuggingFace，檢查網路連接是否正常。

**Q: 生成的語音聽起來不自然？**  
A: 嘗試調整「韻律變化」參數（0.5-1.0 之間），或更換模型。

**Q: 端口 7860 被佔用？**  
A: 修改 `app.py` 最後一行中的 `server_port` 數值。

**Q: 遷移後無法執行？**  
A: 確認已安裝 Python 及依賴套件。若 `piper-tts` 安裝失敗，可嘗試 `pip install piper`。

## 更多資源

- [Piper TTS 官方](https://github.com/rhasspy/piper)
- [模型試聽](https://rhasspy.github.io/piper-samples/)
- [完整模型列表](https://huggingface.co/rhasspy/piper-voices)
