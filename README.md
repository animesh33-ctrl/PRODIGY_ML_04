# Hand Gesture Recognition using Deep Learning

This project implements a **hand gesture recognition model** that can accurately identify and classify different hand gestures from image or video data.  
It enables intuitive **human-computer interaction** and gesture-based control systems.

The model is trained on the [LEAP Gestures Dataset](https://www.kaggle.com/gti-upm/leapgestrecog) from Kaggle.

---

## Features

- Recognizes multiple hand gestures from images or real-time video.
- Uses a **deep learning model** (CNN or other architecture) for gesture classification.
- Supports **webcam-based real-time inference**.
- Shows predicted gesture on live video feed.
- Can be extended for gesture-based control of applications or devices.

---

## Requirements

- Python 3.8+
- Libraries:

```bash
pip install numpy pandas tensorflow keras opencv-python matplotlib scikit-learn joblib
```

## 📂 Project Structure

```bash


File Structure
├── best_resnet18_96.pth [Download](<https://drive.google.com/uc?export=download%26id=1v-55qBSYhXZCd-0SQ-z>) # Trained resnet model
├── best_smallcnn_96.pth           # Trained deep learning model
├── 04_Hand_Gesture.ipynb          # Script to train model on LEAP dataset
├── 04_Hand_Gesture_Webcam.py      # Webcam-based inference script
├── README.md                        # Project documentation

```

## How to Run Webcam Inference

```bash
Ensure your webcam is connected.

Run the inference script:

python webcam_gesture_recognition.py


A window will appear showing the webcam feed with the predicted gesture label overlaid on the video.

```
