# 🧠 Real-time Facial Emotion Recognition using MobileNetV2

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red.svg)

This project is a high-performance, real-time facial emotion recognition system developed as a final project for **Digital Image Processing**. It uses **MobileNetV2** (Transfer Learning) to classify human emotions into 7 distinct categories with high accuracy and speed.



---

## 🌟 Key Features
* **7 Emotion Classes:** Detects Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
* **Hybrid Interface:** Run it as a **Local OpenCV window** or a **Streamlit Web App**.
* **Temporal Smoothing:** Implemented a `deque` buffer to average predictions, making the bounding box labels stable and flicker-free.
* **Dynamic Colors:** Bounding boxes change colors based on the detected emotion (e.g., Green for Happy, Red for Angry).
* **Cross-Platform:** Rebuilt architecture ensures the model loads perfectly on both Keras 2 and Keras 3 environments.

---

## 🛠️ Technology Stack
* **Deep Learning Framework:** TensorFlow & Keras
* **Base Model:** MobileNetV2 (Feature Extractor)
* **Computer Vision:** OpenCV (Haar Cascades for Face Detection)
* **Web UI:** Streamlit
* **Language:** Python 3.11

---

## 📂 Project Structure
* `webcam.py`: The main hybrid script (OpenCV + Streamlit).
* `emotion_model.h5`: Pre-trained weights for the MobileNetV2 model.
* `requirements.txt`: List of necessary Python libraries.
* `haarcascade_frontalface_default.xml`: OpenCV's face detection model.

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/AsadRiaz045/Facial-Emotion-Recognition-AI.git](https://github.com/AsadRiaz045/Facial-Emotion-Recognition-AI.git)
cd Facial-Emotion-Recognition-AI
