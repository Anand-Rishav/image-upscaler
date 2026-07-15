from __future__ import annotations

import re
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from upscale import DEFAULT_MODEL_DIR, REALESRGAN_MODELS, UpscaleOptions, upscale_file


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]


def clean_stem(name: str) -> str:
    stem = Path(name).stem or "image"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "image"


def scale_label(scale: float) -> str:
    return f"x{scale:g}".replace(".", "_")


def unique_path(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = folder / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not create a unique output path.")


def image_details(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        return width, height, image.mode


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {f".{ext}" for ext in SUPPORTED_TYPES}:
        suffix = ".png"

    filename = f"{clean_stem(uploaded_file.name)}_{int(time.time())}{suffix}"
    path = unique_path(UPLOAD_DIR, filename)
    path.write_bytes(uploaded_file.getbuffer())
    return path


def build_options(
    *,
    engine: str,
    model: str,
    opencv_model: str,
    scale: float,
    tile: int,
    face_enhance: bool,
    denoise_strength: float,
    sharpen: bool,
    contrast: bool,
    fp32: bool,
) -> UpscaleOptions:
    return UpscaleOptions(
        scale=scale,
        engine=engine,
        model=model,
        opencv_model=opencv_model,
        model_dir=DEFAULT_MODEL_DIR,
        allow_download=True,
        sharpen=sharpen,
        contrast=contrast,
        tile=tile,
        tile_pad=20,
        pre_pad=0,
        denoise_strength=denoise_strength,
        face_enhance=face_enhance,
        fp32=fp32,
        gpu_id=None,
    )


st.set_page_config(
    page_title="AI Image Upscaler",
    page_icon="up",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
    }
    [data-testid="stMetric"] * {
        color: inherit !important;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Image Upscaler")

with st.sidebar:
    st.subheader("Settings")
    engine = st.selectbox(
        "Engine",
        ["auto", "realesrgan", "opencv", "classical"],
        index=0,
    )
    model = st.selectbox(
        "Real-ESRGAN model",
        sorted(REALESRGAN_MODELS),
        index=sorted(REALESRGAN_MODELS).index("RealESRGAN_x4plus"),
    )
    opencv_model = st.selectbox("OpenCV fallback", ["edsr", "fsrcnn"], index=0)
    scale = st.select_slider("Scale", options=[2.0, 3.0, 4.0], value=4.0)
    tile = st.selectbox("Tile size", [0, 128, 256, 512], index=2)
    output_format = st.selectbox("Output format", ["png", "jpg", "webp"], index=0)

    st.divider()
    face_enhance = st.checkbox("Face enhance", value=False)
    sharpen = st.checkbox("Sharpen", value=True)
    contrast = st.checkbox("Auto contrast", value=False)
    fp32 = st.checkbox("FP32", value=False)
    denoise_strength = st.slider(
        "Denoise strength",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

uploaded = st.file_uploader("Image", type=SUPPORTED_TYPES)
input_path = save_upload(uploaded) if uploaded is not None else None

run = st.button("Upscale", type="primary", use_container_width=True)

if input_path is not None and input_path.exists():
    try:
        width, height, mode = image_details(input_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Input", f"{width} x {height}")
        c2.metric("Target", f"{round(width * scale)} x {round(height * scale)}")
        c3.metric("Mode", mode)
    except Exception as exc:  # noqa: BLE001 - UI should show readable errors.
        st.error(f"Could not read image: {exc}")

if run:
    if input_path is None:
        st.warning("Choose an image first.")
        st.stop()

    if not input_path.exists():
        st.error(f"File not found: {input_path}")
        st.stop()

    output_name = (
        f"{clean_stem(input_path.name)}_upscaled_{scale_label(scale)}.{output_format}"
    )
    output_path = unique_path(OUTPUT_DIR, output_name)
    options = build_options(
        engine=engine,
        model=model,
        opencv_model=opencv_model,
        scale=scale,
        tile=tile,
        face_enhance=face_enhance,
        denoise_strength=denoise_strength,
        sharpen=sharpen,
        contrast=contrast,
        fp32=fp32,
    )

    progress = st.empty()
    progress.info("Upscaling in progress...")
    try:
        engine_used = upscale_file(input_path, output_path, options)
    except Exception as exc:  # noqa: BLE001 - Streamlit boundary.
        progress.empty()
        st.error(f"Upscale failed: {exc}")
        st.stop()

    progress.success(f"Saved: {output_path}")

    original_col, result_col = st.columns(2)
    with original_col:
        st.subheader("Original")
        st.image(str(input_path), use_container_width=True)

    with result_col:
        st.subheader("Upscaled")
        st.image(str(output_path), use_container_width=True)

    out_width, out_height, _mode = image_details(output_path)
    m1, m2, m3 = st.columns(3)
    m1.metric("Output", f"{out_width} x {out_height}")
    m2.metric("Engine", engine_used)
    m3.metric("File size", f"{output_path.stat().st_size / 1024 / 1024:.2f} MB")

    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "webp": "image/webp",
    }[output_format]
    st.download_button(
        "Download",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime=mime,
        use_container_width=True,
    )
