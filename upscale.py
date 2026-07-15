from __future__ import annotations

import argparse
import math
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class OpenCVModelSpec:
    model: str
    scale: int
    url: str
    filename: str


@dataclass(frozen=True)
class RealESRGANModelSpec:
    name: str
    scale: int
    urls: tuple[str, ...]
    filenames: tuple[str, ...]


OPENCV_MODELS: dict[tuple[str, int], OpenCVModelSpec] = {
    ("edsr", 2): OpenCVModelSpec(
        "edsr",
        2,
        "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
        "EDSR_x2.pb",
    ),
    ("edsr", 3): OpenCVModelSpec(
        "edsr",
        3,
        "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb",
        "EDSR_x3.pb",
    ),
    ("edsr", 4): OpenCVModelSpec(
        "edsr",
        4,
        "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "EDSR_x4.pb",
    ),
    ("fsrcnn", 2): OpenCVModelSpec(
        "fsrcnn",
        2,
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
        "FSRCNN_x2.pb",
    ),
    ("fsrcnn", 3): OpenCVModelSpec(
        "fsrcnn",
        3,
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
        "FSRCNN_x3.pb",
    ),
    ("fsrcnn", 4): OpenCVModelSpec(
        "fsrcnn",
        4,
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
        "FSRCNN_x4.pb",
    ),
}


REALESRGAN_MODELS: dict[str, RealESRGANModelSpec] = {
    "RealESRGAN_x4plus": RealESRGANModelSpec(
        "RealESRGAN_x4plus",
        4,
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
            "RealESRGAN_x4plus.pth",
        ),
        ("RealESRGAN_x4plus.pth",),
    ),
    "RealESRGAN_x2plus": RealESRGANModelSpec(
        "RealESRGAN_x2plus",
        2,
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
            "RealESRGAN_x2plus.pth",
        ),
        ("RealESRGAN_x2plus.pth",),
    ),
    "RealESRNet_x4plus": RealESRGANModelSpec(
        "RealESRNet_x4plus",
        4,
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/"
            "RealESRNet_x4plus.pth",
        ),
        ("RealESRNet_x4plus.pth",),
    ),
    "RealESRGAN_x4plus_anime_6B": RealESRGANModelSpec(
        "RealESRGAN_x4plus_anime_6B",
        4,
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
            "RealESRGAN_x4plus_anime_6B.pth",
        ),
        ("RealESRGAN_x4plus_anime_6B.pth",),
    ),
    "realesr-animevideov3": RealESRGANModelSpec(
        "realesr-animevideov3",
        4,
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
            "realesr-animevideov3.pth",
        ),
        ("realesr-animevideov3.pth",),
    ),
    "realesr-general-x4v3": RealESRGANModelSpec(
        "realesr-general-x4v3",
        4,
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
            "realesr-general-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
            "realesr-general-wdn-x4v3.pth",
        ),
        ("realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth"),
    ),
}


@dataclass
class UpscaleOptions:
    scale: float
    engine: str
    model: str
    opencv_model: str
    model_dir: Path
    allow_download: bool
    sharpen: bool
    contrast: bool
    tile: int
    tile_pad: int
    pre_pad: int
    denoise_strength: float
    face_enhance: bool
    fp32: bool
    gpu_id: int | None


class UpscaleError(RuntimeError):
    pass


def positive_scale(value: str) -> float:
    try:
        scale = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scale must be a number") from exc
    if scale <= 1:
        raise argparse.ArgumentTypeError("scale must be greater than 1")
    return scale


def denoise_strength(value: str) -> float:
    try:
        strength = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("denoise strength must be a number") from exc
    if strength < 0 or strength > 1:
        raise argparse.ArgumentTypeError("denoise strength must be between 0 and 1")
    return strength


def pil_resampling_filter() -> int:
    return getattr(Image, "Resampling", Image).LANCZOS


def scale_label(scale: float) -> str:
    return f"x{scale:g}".replace(".", "_")


def is_integer_scale(scale: float) -> bool:
    return math.isclose(scale, round(scale), abs_tol=1e-9)


def decompose_opencv_scale(scale: float) -> list[int] | None:
    if not is_integer_scale(scale):
        return None

    remaining = int(round(scale))
    factors: list[int] = []
    for factor in (4, 3, 2):
        while remaining > 1 and remaining % factor == 0:
            factors.append(factor)
            remaining //= factor

    if remaining != 1:
        return None
    return factors


def download_file(url: str, path: Path, allow_download: bool) -> Path:
    if path.exists():
        return path

    if not allow_download:
        raise UpscaleError(f"Missing model file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading model: {path.name}")
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        temp_path.replace(path)
    except Exception as exc:  # noqa: BLE001 - this is a CLI boundary.
        temp_path.unlink(missing_ok=True)
        raise UpscaleError(f"Could not download {url}: {exc}") from exc
    return path


def split_alpha(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return image.convert("RGB"), None


def combine_alpha(rgb: Image.Image, alpha: Image.Image | None) -> Image.Image:
    if alpha is None:
        return rgb

    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size, pil_resampling_filter())

    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def apply_rgb_filter(image: Image.Image, filter_obj: ImageFilter.Filter) -> Image.Image:
    if image.mode == "RGBA":
        rgb = image.convert("RGB").filter(filter_obj)
        return combine_alpha(rgb, image.getchannel("A"))
    return image.filter(filter_obj)


def apply_autocontrast(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        rgb = ImageOps.autocontrast(image.convert("RGB"))
        return combine_alpha(rgb, image.getchannel("A"))
    if image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    return ImageOps.autocontrast(image)


def postprocess(image: Image.Image, options: UpscaleOptions, neural_used: bool) -> Image.Image:
    result = image
    if options.contrast:
        result = apply_autocontrast(result)

    if options.sharpen:
        mask = (
            ImageFilter.UnsharpMask(radius=0.65, percent=65, threshold=3)
            if neural_used
            else ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=4)
        )
        result = apply_rgb_filter(result, mask)

    return result


def upscale_with_pillow(image: Image.Image, scale: float, options: UpscaleOptions) -> Image.Image:
    source = ImageOps.exif_transpose(image)
    width, height = source.size
    target_size = (round(width * scale), round(height * scale))
    result = source.resize(target_size, pil_resampling_filter())
    return postprocess(result, options, neural_used=False)


def create_open_cv_superres():
    try:
        import cv2
    except ImportError as exc:
        raise UpscaleError(
            "OpenCV is not installed. Install the light stack with: "
            "python -m pip install -r requirements-light.txt"
        ) from exc

    if not hasattr(cv2, "dnn_superres"):
        raise UpscaleError("OpenCV dnn_superres is missing. Install opencv-contrib-python.")

    creator = getattr(cv2.dnn_superres, "DnnSuperResImpl_create", None)
    if creator is not None:
        return cv2, creator()

    impl = getattr(cv2.dnn_superres, "DnnSuperResImpl", None)
    if impl is not None and hasattr(impl, "create"):
        return cv2, impl.create()

    raise UpscaleError("Could not create the OpenCV DNN super-resolution object.")


def opencv_model_path(
    model_name: str,
    factor: int,
    model_dir: Path,
    allow_download: bool,
) -> Path:
    spec = OPENCV_MODELS.get((model_name, factor))
    if spec is None:
        raise UpscaleError(f"No OpenCV model URL for {model_name} x{factor}.")
    return download_file(spec.url, model_dir / "opencv" / spec.filename, allow_download)


def run_opencv_once(image: Image.Image, factor: int, options: UpscaleOptions) -> Image.Image:
    import numpy as np

    cv2, sr = create_open_cv_superres()
    model_path = opencv_model_path(
        options.opencv_model,
        factor,
        options.model_dir,
        options.allow_download,
    )

    rgb, alpha = split_alpha(image)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    sr.readModel(str(model_path))
    sr.setModel(options.opencv_model, factor)
    upscaled_bgr = sr.upsample(bgr)
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)

    return combine_alpha(Image.fromarray(upscaled_rgb), alpha)


def upscale_with_opencv(image: Image.Image, options: UpscaleOptions) -> Image.Image:
    factors = decompose_opencv_scale(options.scale)
    if not factors:
        raise UpscaleError(
            "OpenCV neural mode supports integer scales that can be made from x2, "
            "x3, and x4 passes, for example 2, 3, 4, 6, 8, 12, or 16."
        )

    result = ImageOps.exif_transpose(image)
    for factor in factors:
        result = run_opencv_once(result, factor, options)
    return postprocess(result, options, neural_used=True)


def import_realesrgan_stack():
    try:
        import cv2
        import numpy as np
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    except ImportError as exc:
        raise UpscaleError(
            "Real-ESRGAN dependencies are not installed. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return cv2, np, torch, RRDBNet, RealESRGANer, SRVGGNetCompact


def build_realesrgan_architecture(model_name: str, RRDBNet, SRVGGNetCompact):
    if model_name in {"RealESRGAN_x4plus", "RealESRNet_x4plus"}:
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

    if model_name == "RealESRGAN_x2plus":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )

    if model_name == "RealESRGAN_x4plus_anime_6B":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4,
        )

    if model_name == "realesr-animevideov3":
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )

    if model_name == "realesr-general-x4v3":
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )

    raise UpscaleError(f"Unknown Real-ESRGAN model: {model_name}")


def realesrgan_model_paths(options: UpscaleOptions) -> tuple[str | list[str], list[float] | None]:
    spec = REALESRGAN_MODELS[options.model]
    paths = [
        download_file(url, options.model_dir / "realesrgan" / filename, options.allow_download)
        for url, filename in zip(spec.urls, spec.filenames)
    ]

    if options.model == "realesr-general-x4v3" and options.denoise_strength != 1:
        return [str(paths[0]), str(paths[1])], [
            options.denoise_strength,
            1 - options.denoise_strength,
        ]

    return str(paths[0]), None


def imread_unicode(path: Path):
    cv2, np, *_rest = import_realesrgan_stack()
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise UpscaleError(f"Could not read image: {path}")
    return image


def prepare_cv_image_for_save(image, suffix: str):
    if suffix.lower() not in {".jpg", ".jpeg"}:
        return image

    cv2, np, *_rest = import_realesrgan_stack()
    if image.ndim == 3 and image.shape[2] == 4:
        color = image[:, :, :3].astype("float32")
        alpha = image[:, :, 3:4].astype("float32") / 255.0
        white = np.full_like(color, 255, dtype="float32")
        return (color * alpha + white * (1 - alpha)).clip(0, 255).astype("uint8")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def imwrite_unicode(path: Path, image) -> None:
    cv2, _np, *_rest = import_realesrgan_stack()
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower() or ".png"
    image = prepare_cv_image_for_save(image, suffix)

    params = []
    if suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]

    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise UpscaleError(f"Could not encode output as {suffix}")
    encoded.tofile(str(path))


def upscale_file_with_realesrgan(input_path: Path, output_path: Path, options: UpscaleOptions) -> None:
    cv2, _np, torch, RRDBNet, RealESRGANer, SRVGGNetCompact = import_realesrgan_stack()
    model_spec = REALESRGAN_MODELS[options.model]
    model = build_realesrgan_architecture(options.model, RRDBNet, SRVGGNetCompact)
    model_path, dni_weight = realesrgan_model_paths(options)
    half = torch.cuda.is_available() and not options.fp32

    upsampler = RealESRGANer(
        scale=model_spec.scale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=options.tile,
        tile_pad=options.tile_pad,
        pre_pad=options.pre_pad,
        half=half,
        gpu_id=options.gpu_id,
    )

    image = imread_unicode(input_path)
    if options.face_enhance:
        try:
            from gfpgan import GFPGANer
        except ImportError as exc:
            raise UpscaleError(
                "Face enhancement requires GFPGAN. Install the full requirements.txt."
            ) from exc

        face_enhancer = GFPGANer(
            model_path=(
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
                "GFPGANv1.3.pth"
            ),
            upscale=options.scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        _cropped, _restored, output = face_enhancer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
    else:
        output, _mode = upsampler.enhance(image, outscale=options.scale)

    if options.sharpen or options.contrast:
        output = cv_to_pil_to_cv(output, options)

    imwrite_unicode(output_path, output)


def cv_to_pil_to_cv(image, options: UpscaleOptions):
    cv2, np, *_rest = import_realesrgan_stack()

    if image.ndim == 2:
        pil = Image.fromarray(image, mode="L")
    elif image.shape[2] == 4:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    pil = postprocess(pil, options, neural_used=True)

    if pil.mode == "RGBA":
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGBA2BGRA)
    if pil.mode == "L":
        return np.asarray(pil)
    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def flatten_for_jpeg(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")

    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A"))
    return background


def save_pillow_image(image: Image.Image, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in {".jpg", ".jpeg"}:
        flatten_for_jpeg(image).save(
            output_path,
            quality=95,
            subsampling=0,
            optimize=True,
        )
        return

    if suffix == ".webp":
        image.save(output_path, quality=95, method=6)
        return

    if suffix == ".png":
        image.save(output_path, optimize=True, compress_level=6)
        return

    image.save(output_path)


def upscale_file_with_pillow_engine(
    input_path: Path,
    output_path: Path,
    options: UpscaleOptions,
    engine: str,
) -> str:
    with Image.open(input_path) as image:
        if engine == "opencv":
            upscaled = upscale_with_opencv(image, options)
            engine_used = f"opencv:{options.opencv_model}"
        else:
            upscaled = upscale_with_pillow(image, options.scale, options)
            engine_used = "classical"
        save_pillow_image(upscaled, output_path)
    return engine_used


def upscale_file(input_path: Path, output_path: Path, options: UpscaleOptions) -> str:
    if options.engine == "realesrgan":
        upscale_file_with_realesrgan(input_path, output_path, options)
        return f"realesrgan:{options.model}"

    if options.engine == "opencv":
        return upscale_file_with_pillow_engine(input_path, output_path, options, "opencv")

    if options.engine == "classical":
        return upscale_file_with_pillow_engine(input_path, output_path, options, "classical")

    try:
        upscale_file_with_realesrgan(input_path, output_path, options)
        return f"realesrgan:{options.model}"
    except UpscaleError as exc:
        print(f"Real-ESRGAN unavailable ({exc}); trying OpenCV neural fallback.", file=sys.stderr)

    try:
        return upscale_file_with_pillow_engine(input_path, output_path, options, "opencv")
    except UpscaleError as exc:
        print(f"OpenCV neural fallback unavailable ({exc}); using Lanczos.", file=sys.stderr)
        return upscale_file_with_pillow_engine(input_path, output_path, options, "classical")


def iter_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise UpscaleError(f"Unsupported image type: {source.suffix}")
        return [source]

    if not source.is_dir():
        raise UpscaleError(f"Input path does not exist: {source}")

    pattern: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    files = sorted(
        path
        for path in pattern
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not files:
        raise UpscaleError(f"No supported images found in {source}")
    return files


def default_output_name(input_path: Path, scale: float) -> str:
    return f"{input_path.stem}_upscaled_{scale_label(scale)}.png"


def output_path_for(
    input_path: Path,
    source: Path,
    output: Path | None,
    multiple: bool,
    scale: float,
) -> Path:
    if not multiple:
        if output is None:
            return input_path.with_name(default_output_name(input_path, scale))
        if output.suffix:
            return output
        return output / default_output_name(input_path, scale)

    output_dir = output if output is not None else source / "upscaled"
    if output_dir.suffix:
        raise UpscaleError("Batch mode output must be a directory, not a file.")
    relative = input_path.relative_to(source)
    return output_dir / relative.with_name(default_output_name(input_path, scale))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upscale images with Real-ESRGAN first, OpenCV EDSR second, and "
            "high-quality Lanczos as a final fallback."
        )
    )
    parser.add_argument("input", type=Path, help="Image file or folder to upscale.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for one image, or output folder for batch mode.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=positive_scale,
        default=4.0,
        help="Output scale. Real-ESRGAN supports arbitrary outscale. Default: 4.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "realesrgan", "opencv", "classical"),
        default="auto",
        help="auto tries Real-ESRGAN, then OpenCV, then classical. Default: auto.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(REALESRGAN_MODELS),
        default="RealESRGAN_x4plus",
        help="Real-ESRGAN model. Default: RealESRGAN_x4plus.",
    )
    parser.add_argument(
        "--opencv-model",
        choices=sorted({model for model, _scale in OPENCV_MODELS}),
        default="edsr",
        help="OpenCV fallback model. edsr is higher quality; fsrcnn is faster.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Folder where downloaded model weights are stored.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing model files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a folder, include images in subfolders.",
    )
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        help="Run GFPGAN face restoration after Real-ESRGAN background upscaling.",
    )
    parser.add_argument(
        "--denoise-strength",
        type=denoise_strength,
        default=1.0,
        help="Only used by realesr-general-x4v3. 1 keeps detail, 0 is strongest denoise.",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Tile size for Real-ESRGAN. Use 256 or 512 if GPU memory is low.",
    )
    parser.add_argument(
        "--tile-pad",
        type=int,
        default=20,
        help="Padding around Real-ESRGAN tiles. Default: 20.",
    )
    parser.add_argument(
        "--pre-pad",
        type=int,
        default=0,
        help="Extra padding before Real-ESRGAN inference. Default: 0.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 instead of fp16 on CUDA. Slower, sometimes more stable.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="CUDA GPU id. Omit to let Real-ESRGAN choose.",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable final sharpening.",
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="Apply light auto-contrast after upscaling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source = args.input.resolve()
    output = args.output.resolve() if args.output else None
    options = UpscaleOptions(
        scale=args.scale,
        engine=args.engine,
        model=args.model,
        opencv_model=args.opencv_model,
        model_dir=args.model_dir.resolve(),
        allow_download=not args.no_download,
        sharpen=not args.no_sharpen,
        contrast=args.contrast,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        denoise_strength=args.denoise_strength,
        face_enhance=args.face_enhance,
        fp32=args.fp32,
        gpu_id=args.gpu_id,
    )

    try:
        inputs = iter_images(source, args.recursive)
    except UpscaleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    multiple = len(inputs) > 1 or source.is_dir()
    failures = 0

    for input_path in inputs:
        try:
            out_path = output_path_for(input_path, source, output, multiple, args.scale)
            engine_used = upscale_file(input_path, out_path, options)
        except Exception as exc:  # noqa: BLE001 - batch mode should keep going.
            failures += 1
            print(f"[failed] {input_path}: {exc}", file=sys.stderr)
            continue

        print(f"[ok] {input_path} -> {out_path} ({engine_used})")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
