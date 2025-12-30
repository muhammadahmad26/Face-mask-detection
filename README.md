# 😷 Face Mask Detection using YOLOv8

A complete **end-to-end computer vision project** for **Face Mask Detection**, covering dataset conversion, YOLO format preparation, model training, and real-time webcam inference using **YOLOv8**.

This project demonstrates **practical experience in data preprocessing, object detection training, and deployment-ready inference pipelines**.



## 📌 Project Overview

This project focuses on detecting three face-mask related classes:

* **with_mask**
* **without_mask**
* **mask_weared_incorrect**

The original dataset (VOC XML format) is converted into **YOLO format**, trained using a **pretrained YOLOv8 model**, and tested using **real-time webcam inference**.



## 🧠 Key Skills Demonstrated

* Dataset preprocessing & annotation conversion (VOC → YOLO)
* Train/validation dataset splitting
* Object detection using **YOLOv8**
* Model training with **Ultralytics**
* Real-time inference using **OpenCV**
* Kaggle notebook environment handling
* Production-style ML workflow



## 🗂 Dataset Details

* **Source:** Kaggle Face Mask Detection Dataset
* **Original Format:** Pascal VOC (XML annotations)
* **Converted Format:** YOLO

### Classes Mapping

```text
0 → with_mask
1 → without_mask
2 → mask_weared_incorrect
```


## 🔄 Dataset Conversion Workflow

The following steps are performed programmatically:

1. Read XML annotation files
2. Normalize bounding boxes
3. Convert annotations to YOLO format
4. Split dataset into:

   * 80% Training
   * 20% Validation
5. Organize folder structure compatible with YOLOv8

### Output Directory Structure

```text
dataset_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```



## ⚙️ Dependencies & Installation

Install required libraries:

```bash
pip install ultralytics opencv-python lxml tqdm
```


## 📄 YOLO Dataset Configuration (`data.yaml`)

```yaml
path: /kaggle/working/dataset_yolo
train: /kaggle/working/dataset_yolo/images/train
val: /kaggle/working/dataset_yolo/images/val

nc: 3
names:
  - with_mask
  - without_mask
  - mask_weared_incorrect
```



## 🚀 Model Training

* **Model:** YOLOv8 Small (`yolov8s.pt`)
* **Framework:** Ultralytics YOLO
* **Epochs:** 10
* **Image Size:** 640
* **Batch Size:** 16

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    project="yolo_mask",
    name="exp1",
    exist_ok=True
)
```



## 🎥 Real-Time Webcam Inference

This script runs **live face mask detection** using your trained model:

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Face Mask Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Press **`Q`** to exit.



## 🏆 Results

* Successfully detects:

  * Mask worn correctly
  * Mask not worn
  * Mask worn incorrectly
* Works in **real-time**
* Lightweight & deployable


## 🧑‍💻 Author Experience

**Role:** Computer Vision / Machine Learning Practitioner
**Experience Highlights:**

* Hands-on with YOLO-based detection systems
* Dataset engineering & annotation pipelines
* Kaggle ML workflows
* Real-time inference using OpenCV
* Strong foundation in applied deep learning


## 📚 References & Resources

* Kaggle – Dataset hosting and experimentation
* YOLOv8 – Real-time object detection
* Ultralytics – YOLOv8 framework
* OpenCV – Image & video processing
* Pascal VOC Annotation Format



## 📜 License

This project is intended for **educational and research purposes**.
Dataset license belongs to the original Kaggle dataset provider.



## ⭐ Acknowledgments

Thanks to the **open-source computer vision community** and **Ultralytics YOLO team** for enabling practical AI solutions.
