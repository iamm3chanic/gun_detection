import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

# --- Настройки ---
MODELS = {
    "Detection": {
        "YOLOv8n": "runs/best_yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLOv8m": "yolov8m.pt",
        "Faster R-CNN": "faster_rcnn_R_50_FPN_3x.yaml"  # Пример, нужны веса
    },
    "Segmentation": {
        "YOLOv8s-seg": None,
        "Mask R-CNN": None
    },
    "Classification": {
        "MobileNetV3": None,
        "ResNet18": None
    }
}

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "mp4", "gif", "tiff", "tif"]


# --- Функции ---
def load_model(model_name, type='Detection'):
    if "YOLO" in model_name:
        return YOLO(MODELS[type][model_name])
    else:
        raise ValueError("The model is not supported")


def process_image(model, image):
    try:
        # Если это уже numpy-массив (например, кадр видео)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image_np = np.array(image)

        results = model.predict(image_np)
        annotated_image = results[0].plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image, len(results[0].boxes)
    except Exception as e:
        st.error(f"Error: {e}")
        return None, 0


def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    metrics = {"mAP50": 0.0, "IoU": 0.0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция и расчет метрик (заглушка)
        processed_frame, _ = process_image(model, frame)
        frames.append(processed_frame)

    cap.release()
    return frames, fps, metrics

# --- Обработка ошибок ---
@st.cache_data
def check_file(file):
    if file is None:
        return False
    ext = file.name.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS



# --- Интерфейс ---
uploaded_file = None
st.sidebar.title("Navigation")
with st.sidebar:
    selected_technology = option_menu(
        menu_title='',
        options=['Detection',
                 'Segmentation',
                 'Classification',
                 ]
        , icons=['water', "search", 'hexagon']
        , menu_icon='display'
        , default_index=0)

# tab1, tab2, tab3 = st.tabs(["Detection", "Segmentation", "Classification"])

if selected_technology == 'Detection':
    st.title("Weapon Detector")
    model_name = st.selectbox("Model", list(MODELS["Detection"].keys()))
    uploaded_file = st.file_uploader(
        "Load photo/video",
        type=ALLOWED_EXTENSIONS,
        help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
    )

    if uploaded_file:
        model = load_model(model_name,type='Detection')
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Обработка фото
        if file_ext in ["jpg", "jpeg", "png", "tiff", "tif"]:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            with st.spinner("Analysing..."):
                result_image, detections = process_image(model, image_np)
                st.image(result_image, caption="Result", use_container_width=True)

                if detections == 0:
                    st.warning("Nothing was found! Try a different model.")
                else:
                    st.success(f"Objects found: {detections}")

                    # Конвертируем результат в байты
                    img_bytes = cv2.imencode(".jpg", result_image)[1].tobytes()

                    st.download_button(
                        label="Download result",
                        data=img_bytes,
                        file_name=f"{uploaded_file.name.split('.')[0]}_{model_name}_detection.jpg",
                        mime="image/jpeg"
                    )


        # Обработка видео/GIF
        elif file_ext in ["mp4", "gif"]:
            st.video(uploaded_file)  # Show original

            temp_path = f"temp.{file_ext}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Настройки обработки видео
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            status_text = st.empty()
            summary = {"frames_with_weapons": 0}

            # Временный файл для результата
            output_path = "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Обработка по кадрам
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Детекция и запись кадра
                processed_frame, detections = process_image(model, frame)
                if detections > 0:
                    summary["frames_with_weapons"] += 1
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

                # Обновляем прогресс
                progress = (frame_num + 1) / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processed: {frame_num + 1}/{total_frames} frames")

            st.write(f"**Summary:** Weapons detected in {summary['frames_with_weapons']}/{total_frames} frames")

            cap.release()
            out.release()

            # Показываем результат
            st.video(output_path)

            # Кнопка скачивания
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download video with detection",
                    data=f,
                    file_name=f"{uploaded_file.name.split('.')[0]}_{model_name}_detection.mp4",
                    mime="video/mp4"
                )

            # Удаляем временные файлы
            os.remove(temp_path)
            os.remove(output_path)

    else:
        st.info("Upload the file for analysis")

    if uploaded_file and not check_file(uploaded_file):
        st.error("Unsupported file format!")


elif selected_technology == 'Segmentation':
    st.header("Segmentation")
    model_name = st.selectbox("Model", list(MODELS["Segmentation"].keys()))
    st.info("Work in progress! Coming soon with mask visualization.")

    if uploaded_file:
        st.warning("Segmentation models are still training - check back later!")

elif selected_technology == 'Classification':
    st.header("Classification")
    model_name = st.selectbox("Model", list(MODELS["Classification"].keys()))
    st.info("Work in progress! Will identify weapon types (AK-47, M16, etc).")

    if uploaded_file:
        st.warning("Classification models are being optimized - stay tuned!")


