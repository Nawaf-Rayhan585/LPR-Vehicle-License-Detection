import cv2
import json
import tempfile
import numpy as np
import streamlit as st
import easyocr
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(page_title="Vehicle License Plate Reader", layout="wide")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FRAME_SKIP = 5  # only run OCR every N frames


@st.cache_resource
def load_models():
    vehicle_model = YOLO("yolov8n.pt")
    # EasyOCR: pure Python, no C++ backend issues, works on Python 3.13 + Windows
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return vehicle_model, reader


def run_ocr(reader, img):
    """Run EasyOCR on a crop.
    Yields (text, confidence, box_points_array) where box is shape (4, 2)."""
    try:
        results = reader.readtext(img)
    except Exception as e:
        st.warning(f"OCR failed on crop: {e}")
        return

    for (box, text, conf) in results:
        box = np.array(box, dtype=np.int32)  # shape (4, 2)
        yield text, conf, box


def process_video(video_path, vehicle_model, reader, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    results_log = []
    frame_idx = 0
    seen_plates = set()

    progress = st.progress(0, text="Processing video...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = vehicle_model(frame, verbose=False)[0]

        for box in detections.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                VEHICLE_CLASSES[cls_id],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            if frame_idx % FRAME_SKIP != 0:
                continue

            vehicle_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if vehicle_crop.size == 0:
                continue

            for text, conf, box_pts in run_ocr(reader, vehicle_crop):
                text = text.strip()
                if len(text) < 4 or conf < 0.6:
                    continue  # skip junk / low-confidence reads

                # Offset box back to full-frame coordinates
                plate_box = box_pts + np.array([x1, y1], dtype=np.int32)
                cv2.polylines(
                    frame, [plate_box], isClosed=True, color=(0, 255, 0), thickness=2
                )
                cv2.putText(
                    frame,
                    text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                if text not in seen_plates:
                    seen_plates.add(text)
                    results_log.append(
                        {
                            "plate": text,
                            "confidence": round(float(conf), 2),
                            "vehicle_type": VEHICLE_CLASSES[cls_id],
                            "frame": frame_idx,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

        writer.write(frame)
        progress.progress(min(frame_idx / total_frames, 1.0), text="Processing video...")

    cap.release()
    writer.release()
    progress.empty()
    return results_log


def main():
    st.title("🚗 Vehicle License Plate Reader")
    st.caption(
        "Upload a video. It detects vehicles, reads license plates, and gives you a JSON report."
    )

    uploaded_file = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        vehicle_model, reader = load_models()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name

        output_path = input_path.replace(".mp4", "_out.mp4")

        if st.button("Run Detection"):
            with st.spinner("Detecting vehicles and reading plates..."):
                results = process_video(input_path, vehicle_model, reader, output_path)

            st.success(f"Done. Found {len(results)} unique plate(s).")

            if results:
                st.subheader("Detected License Plates")
                st.table(results)

            json_data = json.dumps(results, indent=2)
            st.download_button(
                "Download JSON Report",
                data=json_data,
                file_name="plate_results.json",
                mime="application/json",
            )

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )

            st.video(output_path)


if __name__ == "__main__":
    main()