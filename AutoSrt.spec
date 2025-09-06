# whisper_gui.spec  — PyInstaller onefile，含 ffmpeg.exe 與 whisper/assets，排除所有 .pt

import os
import sys
import whisper

from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.building.build_main import Analysis, PYZ, EXE, TOC

block_cipher = None

# ============= 輔助：過濾 *.pt（雙保險） =============
def strip_pt_from_toc(toc: TOC) -> TOC:
    return TOC([
        it for it in toc
        if not (
            isinstance(it, (list, tuple)) and len(it) > 0
            and str(it[0]).lower().endswith('.pt')
        )
    ])

# ============= 必要的 datas / binaries =============
datas = []
binaries = []
hiddenimports = ['tqdm.auto']  # tqdm 可能動態載入

# Whisper 的 assets（需要 mel_filters.npz）
whisper_assets = os.path.join(os.path.dirname(whisper.__file__), "assets")
datas.append((whisper_assets, "whisper/assets"))

# ffmpeg.exe — 請調整來源路徑到你的專案位置
# 會放在執行時臨時解壓的根目錄（._MEIPASS）
ffmpeg_src = os.path.join('vendor', 'ffmpeg', 'ffmpeg.exe')
if os.path.exists(ffmpeg_src):
    binaries.append((ffmpeg_src, '.'))
else:
    # 你也可以改成絕對路徑，例如：
    # binaries.append((r"C:\tools\ffmpeg\bin\ffmpeg.exe", '.'))
    # 若找不到就不加，程式會嘗試用 PATH 中的 ffmpeg
    pass

# Torch 的動態庫（僅收 .dll/.pyd 等執行檔庫；不含任何 .pt）
binaries += collect_dynamic_libs('torch')

# ============= Analysis =============
a = Analysis(
    ['AutoSrt.py'],     # ★ 你的主程式檔名
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 雙保險：把可能混進來的 .pt 從 TOC 剔除
a.binaries = strip_pt_from_toc(a.binaries)
a.datas    = strip_pt_from_toc(a.datas)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============= Onefile EXE =============
# onefile 的寫法：直接把 a.binaries / a.zipfiles / a.datas 丟進 EXE
# 不要再建 COLLECT。
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='WhisperSRT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # 若想看 console 訊息改 True
    icon=None
)
