from ultralytics import YOLO
import cv2
import torch
from ultralytics.utils.metrics import box_iou

# Скачиваем датасет (если ещё нет)
#from roboflow import Roboflow
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

PIC1 = 'gun-detection-1fbbu-1/test/images/gun1.png'
PIC2 = 'gun-detection-1fbbu-1/test/images/0b1219fded94d150_jpg.rf.366a43d430767bd0da416b7ab14881ec.jpg'

def download_data(tag='yolov8'):
    rf = Roboflow(api_key="srM3N5mvtqTb1DxLIuW2")
    project = rf.workspace("gun-detection-1lttj").project("gun-detection-1fbbu")
    dataset = project.version(1).download(tag)
    return dataset.location

def train_model(model_name="yolov8n.pt", data_path="gun-detection-1fbbu-1/data.yaml", **kwargs):
    # Загрузка модели
    model = YOLO(model_name)
    # Обучение (если нужно)
    # model.train(data="gun-detection-1fbbu-1/data.yaml", epochs=5)
    # получше - для колаба с Т4
    model.train(
        data=data_path,
        freeze=[0, 1, 2],  # Первые 3 слоя не обучаются
        amp=True,  # Использует FP16 → ускорение на GPU
        epochs=20,  # Уменьшим, если долго (но лучше не меньше 10)
        imgsz=640,  # 320 - если тормозит, но качество упадёт
        batch=16,  # Максимально возможный (для T4 можно 32)
        workers=4,  # Число ядер CPU для загрузки данных
        optimizer="AdamW",  # Быстрее сходится, чем SGD
        lr0=0.001,  # Скорость обучения (можно увеличить до 0.01, если модель не взрывается)
        patience=3,  # Ранняя остановка, если нет улучшений
        **kwargs
    )

def get_trained_model(weights_path="runs/detect/train/weights/best.pt"):
    model = YOLO(weights_path)  # загружаем свои веса
    return model


def predict_and_show(model, image_path, show=True):
    """Предсказывает bbox'ы и показывает результат."""
    results = model.predict(image_path)
    if show:
        results[0].show()  # Для Jupyter/Colab
    return results

def calculate_metrics(model, image_path=PIC2):
    metrics = model.val(data="gun-detection-1fbbu-1/data.yaml", split="val", batch=4, imgsz=320)
    map50 = metrics.box.map50  # mAP@0.5
    print(f"mAP50-95: {metrics.box.map}")
    # Пример вывода IoU для конкретного изображения
    results = model.predict(image_path)
    iou = calculate_iou(results) # Массив IoU для каждого bbox
    print(f"IoU: {iou:.2f}")
    return {"mAP50": map50, "IoU": iou}

def calculate_iou(results):
    """Считает средний IoU для всех bbox'ов на изображении."""
    boxes = results[0].boxes
    if len(boxes) < 2:
        return 0.0
    iou_matrix = box_iou(boxes.xyxy, boxes.xyxy)
    return iou_matrix.mean().item()


import cv2
import time


def process_video(model, input_path="video-gun.mp4", output_path="output.mp4"):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        start_time = time.time()  # Старт таймера для FPS
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция
        results = model.predict(frame)
        annotated_frame = results[0].plot()

        # Расчет FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Запись кадра
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


# Пример использования:
model = get_trained_model()
process_video(model, "input.mp4")


def save_results_with_plot(results, image_path, output_file="result.png", metrics=None):
    # Рисуем bbox'ы
    img = np.array(Image.open(image_path))
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # Добавляем bbox'ы
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2))
        ax.text(x1, y1, f"Conf: {box.conf.item():.2f}", color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.5))
    if metrics:
        ax.text(10, 30, f"mAP50: {metrics['mAP50']:.2f}", color="white", fontsize=12,
                bbox=dict(facecolor="green", alpha=0.7))
    # Сохраняем график
    plt.axis("off")
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    model = get_trained_model()
    # res = predict_and_show(model, PIC1)
    # metrics = calculate_metrics(model, PIC2)
    # print(f"Metrics: {metrics}")

    results = model.predict(PIC1)
    save_results_with_plot(results, PIC1)

    process_video(model)


