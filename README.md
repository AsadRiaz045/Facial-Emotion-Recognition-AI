# 🧠 Real-time Facial Emotion Recognition using

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://facial-emotion-recognition-ai-jndp2fzqdhbessyy5j44ry.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

This project is a high-performance facial emotion recognition system developed as a final project for **Digital Image Processing**. It utilizes **MobileNetV2** (Transfer Learning) to classify 7 distinct human emotions in real-time through a web browser or local webcam.

## 🚀 Live Demo
Experience the project live in your browser:  
👉 **[Live Emotion Detector App](https://facial-emotion-recognition-ai-jndp2fzqdhbessyy5j44ry.streamlit.app/)**

---

## 🌟 Key Features
* **7 Emotion Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
* **Hybrid Interface:** Built to run as a local Python script or a Cloud-deployed Streamlit Web App.
* **WebRTC Integration:** Uses `streamlit-webrtc` for seamless real-time video streaming in the browser.
* **Optimized Architecture:** Rebuilt MobileNetV2 backbone to ensure compatibility across different Keras versions.
* **Visual Feedback:** Dynamic color-coded bounding boxes for each emotion (e.g., Green for Happy, Red for Angry).

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (MobileNetV2)
* **Computer Vision:** OpenCV (Haar Cascades for face detection)
* **Web Framework:** Streamlit & Streamlit-WebRTC
* **Data Processing:** NumPy & Python

---

## 📂 Project Structure
* `webcam.py`: Main hybrid application script.
* `emotion_model.h5`: Pre-trained model weights.
* `requirements.txt`: Configuration for Cloud deployment.
* `README.md`: Project documentation.

## 📦 Local Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AsadRiaz045/Facial-Emotion-Recognition-AI.git](https://github.com/AsadRiaz045/Facial-Emotion-Recognition-AI.git)
   cd Facial-Emotion-Recognition-AI
##2.Install Dependencies
    pip install -r requirements.txt
##3.Run Locally
      python webcam.py
