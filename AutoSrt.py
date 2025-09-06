import sys
import os
import io
import shutil
import subprocess
import datetime
import threading
import time

import torch
import whisper
import srt
import tqdm as tqdm_module  # 只用於轉錄進度（下載改用檔案監看器）

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox,
    QProgressBar, QMessageBox, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal

# 1) 關閉 tqdm 的 console 輸出（torch/whisper 內部的 tqdm 也會尊重這個）
os.environ.setdefault("TQDM_DISABLE", "1")

# 2) 在 GUI exe（console=False）時，stderr/stdout 會是 None；補成記憶體緩衝
if sys.stderr is None:
    sys.stderr = io.StringIO()
if sys.stdout is None:
    sys.stdout = io.StringIO()


# ---------- ffmpeg 尋找（支援被打包後） ----------
def get_ffmpeg_path():
    """
    回傳 ffmpeg 可執行檔路徑：
    1) 先找系統 PATH
    2) 若為 PyInstaller onefile/onedir，找 _MEIPASS 或 exe 同層的 ffmpeg.exe
    """
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    base_dir = getattr(sys, "_MEIPASS", None)
    if base_dir:
        cand = os.path.join(base_dir, "ffmpeg.exe")
        if os.path.exists(cand):
            return cand
    exe_dir = os.path.dirname(sys.executable if getattr(
        sys, "frozen", False) else os.path.abspath(__file__))
    cand = os.path.join(exe_dir, "ffmpeg.exe")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError("找不到 ffmpeg，請確認已打包或已在 PATH 中。")


# ---------- 抽音 ----------
def extract_audio(input_file, output_file="temp_audio.wav"):
    """
    使用 ffmpeg 從影片檔案中抽取音訊，轉成 16kHz/mono wav。
    若已是音訊檔，直接回傳原檔。
    回傳: (audio_path, need_delete_temp: bool)
    """
    ext = os.path.splitext(input_file)[1].lower()
    audio_exts = [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"]
    if ext in audio_exts:
        return input_file, False

    cmd = [
        get_ffmpeg_path(), "-y", "-i", input_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_file
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg 抽音失敗，請確認 ffmpeg 可執行。")
    return output_file, True


# ---------- 近似模型大小（用於下載%估算） ----------
MODEL_SIZE_BYTES = {
    "tiny":     75572083,    # ~72MB
    "base":     145262807,   # ~139MB
    "small":    483617219,   # ~461MB
    "medium":   1528008539,  # ~1.5GB
    "large-v3": 3087371615,  # ~3.0GB（可依實測微調）
}


# ---------- 轉錄執行緒（下載用檔案監看器；轉錄用 tqdm monkey-patch） ----------
class TranscribeThread(QThread):
    download_progress = Signal(int, str)    # (百分比, 訊息)
    transcribe_progress = Signal(int, str)  # (百分比, 訊息)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_name, input_media, output_srt_path, decode_options, cache_dir="./"):
        super().__init__()
        self.model_name = model_name
        self.input_media = input_media
        self.output_srt_path = output_srt_path
        self.decode_options = decode_options
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._phase = "idle"  # "download" | "transcribe" | "idle"

        # 下載監看器
        self._watch_stop = threading.Event()
        self._watch_thread = None

    # ---- 下載進度：用檔案監看器 + MODEL_SIZE_BYTES ----
    def _start_download_watcher(self):
        expect = MODEL_SIZE_BYTES.get(self.model_name)  # 可能 None
        model_basename = f"{self.model_name}.pt"

        def find_candidates():
            names = []
            try:
                for fn in os.listdir(self.cache_dir):
                    lower = fn.lower()
                    if (self.model_name in lower) and (
                        lower.endswith(".pt") or lower.endswith(".tmp")
                        or lower.endswith(".part") or lower.endswith(".download")
                    ):
                        names.append(os.path.join(self.cache_dir, fn))
            except Exception:
                pass
            target = os.path.join(self.cache_dir, model_basename)
            if target not in names:
                names.append(target)
            return names

        def watcher():
            last_bytes = -1
            while not self._watch_stop.is_set():
                cands = find_candidates()
                sizes = [os.path.getsize(p)
                         for p in cands if os.path.exists(p)]
                cur = max(sizes) if sizes else 0

                if expect:
                    pct = int(min(cur / expect, 1.0) * 100)
                    msg = f"{cur/1048576:.2f}/{expect/1048576:.2f} MB（{pct}%）"
                else:
                    pct = 0
                    msg = f"{cur/1048576:.2f} MB"

                if cur != last_bytes:
                    self.download_progress.emit(pct, f"模型下載中：{msg}")
                    last_bytes = cur

                time.sleep(0.2)  # 200ms 輪詢

        self._watch_thread = threading.Thread(target=watcher, daemon=True)
        self._watch_thread.start()

    def _stop_download_watcher(self, final_msg=None):
        if self._watch_thread:
            self._watch_stop.set()
            self._watch_thread.join(timeout=1.0)
            self._watch_thread = None
            self._watch_stop.clear()
        if final_msg:
            self.download_progress.emit(100, final_msg)

    def run(self):
        orig_tqdm = tqdm_module.tqdm  # 只給「轉錄」用
        try:
            # 1) 準備輸入
            audio_file, need_delete = extract_audio(self.input_media)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 2) 先檢查模型是否已存在
            model_path = os.path.join(self.cache_dir, f"{self.model_name}.pt")
            need_download = not os.path.exists(model_path)

            # 3) 下載：有需要才啟動監看器；無需再靠 tqdm 顯示下載
            if need_download:
                self._phase = "download"
                self.download_progress.emit(
                    0, f"未找到 {self.model_name}.pt，開始下載…")
                self._start_download_watcher()
            else:
                self.download_progress.emit(100, f"已找到模型：{model_path}")

            # 4) —— Whisper 自動下載/載入模型 ——（若 need_download=True，監看器會顯示進度）
            model = whisper.load_model(
                name=self.model_name,
                download_root=self.cache_dir,
                device=device
            )

            if need_download:
                self._stop_download_watcher(final_msg="模型下載完成")

            # 5) —— 轉錄（這裡才用 tqdm monkey-patch 顯示進度）——
            self._phase = "transcribe"

            def qt_tqdm(*args, **kwargs):
                # 不往終端輸出
                if "file" not in kwargs or kwargs["file"] is None:
                    kwargs["file"] = io.StringIO()
                kwargs.setdefault("leave", False)

                bar = orig_tqdm(*args, **kwargs)
                total = bar.total or 0

                old_update = bar.update

                def new_update(n=1):
                    old_update(n)
                    if total:
                        pct = int(min(bar.n / total, 1.0) * 100)
                        msg = f"{bar.n}/{total}"
                    else:
                        pct = 0
                        msg = f"{bar.n}"
                    self.transcribe_progress.emit(pct, f"轉錄中：{msg}")
                bar.update = new_update
                return bar

            tqdm_module.tqdm = qt_tqdm

            self.transcribe_progress.emit(0, "開始轉錄音訊…")
            result = model.transcribe(
                audio=audio_file,
                language=self.decode_options["language"],
                verbose=False,
                initial_prompt=self.decode_options["initial_prompt"],
                temperature=self.decode_options["temperature"]
            )

            # 6) 依 segments 產生字幕並匯出（第二層進度）
            segments = result.get("segments", []) or []
            total_segs = max(len(segments), 1)
            subs = []
            for idx, seg in enumerate(segments, start=1):
                start = datetime.timedelta(seconds=seg["start"])
                end = datetime.timedelta(seconds=seg["end"])
                text = seg["text"]
                subs.append(srt.Subtitle(
                    index=idx, start=start, end=end, content=text))
                pct = int(idx / total_segs * 100)
                self.transcribe_progress.emit(pct, f"整理字幕：{idx}/{total_segs}")

            with open(self.output_srt_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(subs))

            if need_delete and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except Exception:
                    pass

            self.transcribe_progress.emit(100, "完成")
            self.finished.emit(self.output_srt_path)

        except Exception as e:
            self.failed.emit(str(e))
        finally:
            # 還原 tqdm（避免影響下一次）
            tqdm_module.tqdm = orig_tqdm
            self._phase = "idle"
            # 萬一監看器還在跑，保險關閉
            self._stop_download_watcher()


# ---------- GUI ----------
class WhisperGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper 字幕轉換器 (PySide6)")
        self.setGeometry(200, 200, 700, 540)

        main = QVBoxLayout()

        # 輸入檔
        row_in = QHBoxLayout()
        self.edit_in = QLineEdit()
        btn_in = QPushButton("選擇檔案")
        btn_in.clicked.connect(self.choose_input_file)
        row_in.addWidget(self.edit_in)
        row_in.addWidget(btn_in)

        # 輸出資料夾
        row_outdir = QHBoxLayout()
        self.edit_outdir = QLineEdit()
        btn_outdir = QPushButton("選擇資料夾")
        btn_outdir.clicked.connect(self.choose_output_dir)
        row_outdir.addWidget(self.edit_outdir)
        row_outdir.addWidget(btn_outdir)

        # 輸出檔名（同名預設）
        self.edit_suffix = QLineEdit()

        # 模型
        self.combo_model = QComboBox()
        self.combo_model.addItems(
            ["tiny", "base", "small", "medium", "large-v3"])
        self.combo_model.setCurrentText("medium")

        # 語言 / 初始提示
        self.edit_lang = QLineEdit("zh")
        self.edit_prompt = QLineEdit("請用繁體中文")

        # temperature
        self.lbl_temp = QLabel("Temperature: 0.0")
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setMinimum(0)
        self.slider_temp.setMaximum(10)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(self.on_temp_changed)

        # 模型下載進度
        self.lbl_dl_title = QLabel("模型下載進度：")
        self.bar_dl = QProgressBar()
        self.lbl_dl_status = QLabel("等待中…")

        # 轉錄進度
        self.lbl_tr_title = QLabel("字幕轉換進度：")
        self.bar_tr = QProgressBar()
        self.lbl_tr_status = QLabel("等待中…")

        # 開始
        btn_start = QPushButton("開始轉換")
        btn_start.clicked.connect(self.start_run)

        # 版面
        main.addWidget(QLabel("輸入檔案："))
        main.addLayout(row_in)
        main.addWidget(QLabel("輸出資料夾："))
        main.addLayout(row_outdir)
        main.addWidget(QLabel("輸出檔名（自動加 .srt）："))
        main.addWidget(self.edit_suffix)
        main.addWidget(QLabel("Model："))
        main.addWidget(self.combo_model)
        main.addWidget(QLabel("Language："))
        main.addWidget(self.edit_lang)
        main.addWidget(QLabel("Initial Prompt："))
        main.addWidget(self.edit_prompt)
        main.addWidget(self.lbl_temp)
        main.addWidget(self.slider_temp)

        main.addWidget(self.lbl_dl_title)
        main.addWidget(self.bar_dl)
        main.addWidget(self.lbl_dl_status)
        main.addWidget(self.lbl_tr_title)
        main.addWidget(self.bar_tr)
        main.addWidget(self.lbl_tr_status)

        main.addWidget(btn_start)
        self.setLayout(main)

    # handlers
    def on_temp_changed(self, *_):
        val = self.slider_temp.value() / 10
        self.lbl_temp.setText(f"Temperature: {val:.1f}")

    def choose_input_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "選擇檔案", "",
            "Media Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.mp4 *.wmv *.avi)"
        )
        if file:
            self.edit_in.setText(file)
            # 預設輸出資料夾與檔名 = 輸入位置與檔名（去副檔名）
            folder = os.path.dirname(file)
            basename = os.path.splitext(os.path.basename(file))[0]
            self.edit_outdir.setText(folder)
            self.edit_suffix.setText(basename)

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if folder:
            self.edit_outdir.setText(folder)

    def start_run(self):
        input_file = self.edit_in.text().strip()
        outdir = self.edit_outdir.text().strip()
        suffix = self.edit_suffix.text().strip()

        if not input_file or not os.path.exists(input_file):
            QMessageBox.critical(self, "錯誤", "請選擇有效的輸入檔案。")
            return
        if not outdir:
            QMessageBox.critical(self, "錯誤", "請選擇輸出資料夾。")
            return
        if not suffix:
            QMessageBox.critical(self, "錯誤", "請輸入輸出檔名。")
            return

        model_name = self.combo_model.currentText()
        language = self.edit_lang.text().strip() or "zh"
        initial_prompt = self.edit_prompt.text().strip() or "請用繁體中文"
        temperature = self.slider_temp.value() / 10

        output_srt = os.path.join(outdir, f"{suffix}.srt")
        decode_options = {
            "language": language,
            "initial_prompt": initial_prompt,
            "temperature": temperature
        }

        self.bar_dl.setValue(0)
        self.lbl_dl_status.setText("等待中…")
        self.bar_tr.setValue(0)
        self.lbl_tr_status.setText("等待中…")

        self.worker = TranscribeThread(
            model_name=model_name,
            input_media=input_file,
            output_srt_path=output_srt,
            decode_options=decode_options,
            cache_dir="./"
        )
        self.worker.download_progress.connect(self.on_download_progress)
        self.worker.transcribe_progress.connect(self.on_transcribe_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    # slots
    def on_download_progress(self, pct, msg):
        self.bar_dl.setValue(pct)
        self.lbl_dl_status.setText(msg)

    def on_transcribe_progress(self, pct, msg):
        self.bar_tr.setValue(pct)
        self.lbl_tr_status.setText(msg)

    def on_finished(self, output_path):
        self.bar_tr.setValue(100)
        self.lbl_tr_status.setText("完成！")
        QMessageBox.information(self, "完成", f"字幕已儲存：\n{output_path}")

    def on_failed(self, err):
        QMessageBox.critical(self, "錯誤", f"處理失敗：\n{err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = WhisperGUI()
    w.show()
    sys.exit(app.exec())
