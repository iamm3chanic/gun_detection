import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

# --- Настройки ---
MODELS = {
    "YOLOv8n": "runs/best/yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "Faster R-CNN (COCO)": "faster_rcnn_R_50_FPN_3x.yaml"  # Пример, нужны веса
}
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "mp4", "gif", "tiff", "tif"]


# --- Функции ---
def load_model(model_name):
    if "YOLO" in model_name:
        return YOLO(MODELS[model_name])
    else:
        raise ValueError("Модель не поддерживается")


def process_image(model, image):
    # Если загруженный файл — это PIL.Image
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image_np = np.array(image)
    # Если это уже numpy-массив (например, кадр видео)
    else:
        image_np = image

    results = model.predict(image_np)
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image, len(results[0].boxes)


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


# --- Интерфейс ---
st.title("🔫 Weapon Detector")
model_name = st.radio("Выберите модель:", list(MODELS.keys()))
uploaded_file = st.file_uploader(
    "Загрузите фото/видео",
    type=ALLOWED_EXTENSIONS,
    help=f"Поддерживаемые форматы: {', '.join(ALLOWED_EXTENSIONS)}"
)

if uploaded_file:
    model = load_model(model_name)
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # Обработка фото
    if file_ext in ["jpg", "jpeg", "png", "tiff", "tif"]:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        with st.spinner("Анализируем..."):
            result_image, detections = process_image(model, image_np)
            st.image(result_image, caption="Результат", use_column_width=True)

            if detections == 0:
                st.warning("Ничего не найдено! Попробуйте другую модель.")
            else:
                st.success(f"Найдено объектов: {detections}")

                # Конвертируем результат в байты
                img_bytes = cv2.imencode(".jpg", result_image)[1].tobytes()

                st.download_button(
                    label="Скачать результат",
                    data=img_bytes,
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )


    # Обработка видео/GIF
    elif file_ext in ["mp4", "gif"]:
        temp_path = f"temp.{file_ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Настройки обработки видео
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        status_text = st.empty()

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
            processed_frame, _ = process_image(model, frame)
            out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

            # Обновляем прогресс
            progress = (frame_num + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Обработано: {frame_num + 1}/{total_frames} кадров")

        cap.release()
        out.release()

        # Показываем результат
        st.video(output_path)

        # Кнопка скачивания
        with open(output_path, "rb") as f:
            st.download_button(
                label="Скачать видео с детекцией",
                data=f,
                file_name="detected_video.mp4",
                mime="video/mp4"
            )

        # Удаляем временные файлы
        os.remove(temp_path)
        os.remove(output_path)

else:
    st.info("Загрузите файл для анализа")


# --- Обработка ошибок ---
@st.cache_data
def check_file(file):
    if file is None:
        return False
    ext = file.name.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS


if uploaded_file and not check_file(uploaded_file):
    st.error("Неподдерживаемый формат файла!")