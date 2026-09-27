# 🚘 LPR — Vehicle License Plate Recognition

Upload a traffic video, get every vehicle detected and every license plate read — plus a JSON report you can download.

![Demo](https://github.com/user-attachments/assets/37a19e8f-0233-48a5-bb06-79e75d30186e)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-purple.svg)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- Detects **cars, motorcycles, buses and trucks** with YOLOv8
- Reads plates with **EasyOCR** (CPU is fine, no GPU needed)
- Draws plate boxes + text on the output video
- Keeps only **unique plates**, with timestamps
- One-click **JSON report** download
- Runs OCR every 5th frame to stay fast

## How it works

```
video ─► YOLOv8 (vehicles) ─► crop each vehicle ─► EasyOCR ─► unique plates ─► annotated video + JSON
```

## Setup

```bash
git clone https://github.com/Nawaf-Rayhan585/LPR-Vehicle-License-Detection.git
cd LPR-Vehicle-License-Detection
pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run main.py
```

Open **http://localhost:8501**, upload a video, and wait for the report.

## Tuning

In `main.py`:

- `FRAME_SKIP` — run OCR every N frames (lower = more accurate, slower)
- `VEHICLE_CLASSES` — which COCO classes count as vehicles

## Want better plate accuracy?

This version reads text from the whole vehicle crop. For production, add a
dedicated plate detector first — there's a trainable one in
[YOLO_Projects/license-plate-recognition](https://github.com/Nawaf-Rayhan585/YOLO_Projects/tree/main/license-plate-recognition).

## License

MIT — see [LICENSE](LICENSE).

Built by **Nawaf Rayhan** · [fayaz7rg@gmail.com](mailto:fayaz7rg@gmail.com) · ⭐ the repo if it helped you
