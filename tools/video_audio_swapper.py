import os
import re
import subprocess
import tempfile

import streamlit as st
import imageio_ffmpeg

# Background audio file sits at the project root (one level up from tools/).
AUDIO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "background_audio.mp3"
)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def get_duration(path: str) -> float:
    """
    Get duration in seconds by parsing ffmpeg's own stderr output.
    (imageio-ffmpeg only bundles ffmpeg, not ffprobe, so we can't shell out
    to ffprobe here - ffmpeg -i on its own reports duration to stderr.)
    """
    result = subprocess.run([FFMPEG_EXE, "-i", path], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError("Could not determine video duration - the uploaded file may be invalid.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def process_video(input_path: str, output_path: str):
    """
    Fast path: copy the original video stream as-is (no re-encoding) and only
    swap/encode the audio track. This is dramatically faster than re-encoding
    the whole video, since video encoding is by far the slow part.
    """
    video_dur = get_duration(input_path)

    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_path,
        "-stream_loop", "-1", "-i", AUDIO_PATH,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",          # <- no video re-encode: this is the speed win
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(video_dur),
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr else "ffmpeg failed")


def run():
    st.title("🎬 Video Audio Swapper")
    st.write("Upload a video and get it back with the background track added.")

    if not os.path.isfile(AUDIO_PATH):
        st.error(f"Background audio file not found in the app folder: {AUDIO_PATH}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a video", type=["mp4", "mov", "mkv", "avi", "webm"], key="video_swapper_uploader"
    )

    if uploaded_file is not None:
        if st.button("Process Video", key="video_swapper_button"):
            with st.spinner("Processing... this can take a minute for longer videos."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                    input_path = os.path.join(tmp_dir, f"input{ext}")
                    output_path = os.path.join(tmp_dir, f"output{ext}")

                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        process_video(input_path, output_path)
                        with open(output_path, "rb") as f:
                            result_bytes = f.read()

                        st.success("Done!")
                        st.download_button(
                            label="Download processed video",
                            data=result_bytes,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_with_audio{ext}",
                            mime="video/mp4",
                            key="video_swapper_download",
                        )
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
