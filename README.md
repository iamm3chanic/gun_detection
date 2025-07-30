# 🔫 Weapon Detection Interface
*Detect guns in images/videos with YOLOv8, MobileNet, Faster R-CNN and Streamlit*  

## 🎯 Features  
- **Detection**: Bounding boxes for weapons.  
- **Segmentation**: Mask support (YOLOv8-seg).  
- **Classification**: In development, supports a few weapon classes.  
- **User-Friendly**: Upload files, view and download results.  

## 📸 Screenshots  
- Detection                                                                           

![Detection](https://github.com/iamm3chanic/gun_detection/screenshots/detection.png) 

- Segmentation

![Segmentation](https://github.com/iamm3chanic/gun_detection/screenshots/segmentation.png)   

- Classification

![Classification](https://github.com/iamm3chanic/gun_detection/screenshots/classification.png)   

## Video

[![Watch the detection video](https://github.com/iamm3chanic/gun_detection/screenshots/video_preview.png)](https://github.com/iamm3chanic/gun_detection/screenshots/video_gun_36sec_YOLOv8n_detection)


## 📊 Metrics  
| Model      | mAP50 | Inference Speed (FPS) |  
|------------|-------|-----------------------|  
| YOLOv8n    | 0.78  | 45                    |  
| YOLOv8s    | 0.80  | 56                    |  
| YOLOv8m    | 0.80  | 60                    |  
| YOLOv8-seg | 0.72  | 30                    |  

## 🛠 Installation  
1. Clone the repo:  
   ```bash
   git clone https://github.com/iamm3chanic/gun_detection.git
   ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Run:

    ```bash
    streamlit run gui.py
    ```
## 📦 Requirements

 ```bash
      Python ≥ 3.11
```

## Libraries:

```bash
streamlit
streamlit_option_menu
numpy>=1.26.1
pandas
ultralytics
torch
roboflow
opencv-python==4.8.0
Pillow
onnxruntime==1.22.1
```

## 📚 Datasets Used

1. [Roboflow Gun Detection](https://universe.roboflow.com/gun-detection-1lttj/gun-detection-1fbbu)

2. [Roboflow Gun Segmentation](https://app.roboflow.com/llm1/guns-segmentation-gekmz/1)

3. [Roboflow Gun Classification](https://universe.roboflow.com/project-tyaeb/gun-classification)

## ⚠️ Known Issues
Segmentation may fail on small objects.

Video processing is slow on CPU.

False-positive detections.

