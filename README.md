# AI Image Upscaler

A Streamlit web app and Python CLI for image upscaling with Real-ESRGAN,
OpenCV neural super-resolution fallback, and classical Lanczos fallback.

The web app is upload-only, so it is suitable for hosting behind a public
domain. Users upload an image in the browser, choose settings, upscale it, and
download the result.

## Setup

Use Python 3.11.

```powershell
cd "D:\image_upscaler"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run The Website

```powershell
cd "D:\image_upscaler"
.\.venv\Scripts\Activate.ps1
streamlit run .\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

For deployment, run the same Streamlit app on your server and point your domain
or reverse proxy to the Streamlit port.

## Streamlit Community Cloud

Deploy with Python 3.11. The Real-ESRGAN dependency stack does not install on
Python 3.14.

When creating the Streamlit Cloud app:

1. Set the repository branch to `main`.
2. Set the main file path to `streamlit_app.py`.
3. Open `Advanced settings`.
4. Set Python version to `3.11`.
5. Deploy the app.

If the app was already created with the wrong Python version, delete it and
redeploy it with Python 3.11 selected.

This repository includes `packages.txt` with `libgl1` for Linux/OpenCV support.
Do not add `libglib2.0-0`; it can fail to resolve on Streamlit Cloud's current
Linux image.

## CLI Usage

Upscale one image with the best available engine:

```powershell
python .\upscale.py "C:\path\to\image.jpg" --scale 4 --tile 256
```

Choose Real-ESRGAN directly:

```powershell
python .\upscale.py "C:\path\to\image.jpg" --engine realesrgan --model RealESRGAN_x4plus --scale 4
```

Enhance faces:

```powershell
python .\upscale.py "C:\path\to\portrait.jpg" --face-enhance --scale 4
```

Use the anime/illustration model:

```powershell
python .\upscale.py "C:\path\to\art.png" --model RealESRGAN_x4plus_anime_6B --scale 4
```

Batch upscale a folder:

```powershell
python .\upscale.py "C:\path\to\folder" --recursive --output ".\outputs" --scale 4 --tile 256
```

## Settings

- `auto` engine tries Real-ESRGAN, then OpenCV, then classical resizing.
- `realesrgan` gives the best AI quality but is slower on CPU.
- `opencv` is a lighter neural fallback.
- `classical` is fast Lanczos resizing without AI detail generation.
- `tile 256` is a good default for memory safety.
- `tile 128` can help when large images fail.
- `tile 0` processes the whole image at once.

## Model Notes

- `RealESRGAN_x4plus` is the default general-purpose model.
- `RealESRGAN_x2plus` is useful when 4x is too much.
- `RealESRNet_x4plus` is cleaner and less hallucinated, but less punchy.
- `RealESRGAN_x4plus_anime_6B` is better for anime, art, and line drawings.
- `realesr-general-x4v3` supports denoise strength.

Outputs default to PNG to avoid adding JPEG compression artifacts.
