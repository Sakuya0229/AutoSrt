import whisper
import srt
import datetime
import time
import torch
import subprocess
import os
from tqdm import tqdm


input_media_name = "去妳妹的撩妹.mp4"  # 可以是 mp3 / mp4 / wmv / avi ...

print(
    f"Now, we are going to transcribe the media file:  ({input_media_name}).")


def extract_audio(input_file, output_file="temp_audio.wav"):
    """
    使用 ffmpeg 從影片檔案中抽取音訊，轉換成 wav。
    如果輸入已經是音訊檔案 (mp3/wav/flac 等)，則直接回傳原始路徑。
    """
    ext = os.path.splitext(input_file)[1].lower()
    audio_exts = [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"]

    if ext in audio_exts:
        return input_file, False  # 已經是音訊檔案，不需要刪除

    print(f"Extracting audio from {input_file} ...")
    # ffmpeg 指令：轉成單聲道、16kHz、wav
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    return output_file, True  # 回傳暫存檔路徑 + 需要刪除標記


def speech_recognition(model_name, input_media, output_subtitle_path, decode_options, cache_dir="./"):
    '''
        Convert audio/video to subtitle.
    '''

    # Record the start time.
    start_time = time.time()

    # 檢查輸入，如果是影片，先抽音訊
    audio_file, need_delete = extract_audio(input_media)

    print(f"=============== Loading Whisper-{model_name} ===============")

    # 判斷是否有 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model = whisper.load_model(
        name=model_name, download_root=cache_dir, device=device)

    print(f"Begin to utilize Whisper-{model_name} to transcribe the audio.")

    # Transcribe the audio.
    transcription = model.transcribe(
        audio=audio_file,
        language=decode_options["language"],
        verbose=False,
        initial_prompt=decode_options["initial_prompt"],
        temperature=decode_options["temperature"]
    )

    # Record the end time.
    end_time = time.time()
    print(
        f"The process of speech recognition costs {end_time - start_time} seconds.")

    subtitles = []
    for i, segment in tqdm(enumerate(transcription["segments"])):
        start_time = datetime.timedelta(seconds=segment["start"])
        end_time = datetime.timedelta(seconds=segment["end"])
        text = segment["text"]
        subtitles.append(srt.Subtitle(
            index=i, start=start_time, end=end_time, content=text))

    srt_content = srt.compose(subtitles)

    print(
        f"\n=============== Saving the subtitle to {output_subtitle_path} ===============")

    with open(output_subtitle_path, "w", encoding="utf-8") as file:
        file.write(srt_content)

    # 如果有暫存音訊，結束後刪除
    if need_delete and os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"Temporary file {audio_file} deleted.")


# === 參數設定 ===
model_name = "medium"  # ["tiny", "base", "small", "medium", "large-v3"]
suffix = "信號與人生"
output_subtitle_path = f"./output-{suffix}.srt"
cache_dir = "./"
language = "zh"
initial_prompt = "請用繁體中文"
temperature = 0

decode_options = {
    "language": language,
    "initial_prompt": initial_prompt,
    "temperature": temperature
}

print(
    f"Setting: (1) Model: whisper-{model_name} (2) Language: {language} (3) Initial Prompt: {initial_prompt} (4) Temperature: {temperature}")
print(f"Transcribe {input_media_name}")

speech_recognition(
    model_name=model_name,
    input_media=input_media_name,
    output_subtitle_path=output_subtitle_path,
    decode_options=decode_options,
    cache_dir=cache_dir
)
