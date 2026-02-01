import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# --- 1. MODEL ARCHITECTURE & LOADING ---
@st.cache_resource
def load_emotion_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(48, 48, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('emotion_model.h5')
    return model

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Colors for bounding boxes (RGB for Streamlit/WebRTC)
color_map = {
    'Angry': (255, 0, 0), 'Disgust': (0, 255, 128), 'Fear': (128, 0, 128),
    'Happy': (0, 255, 0), 'Neutral': (200, 200, 200), 'Sad': (0, 0, 255),
    'Surprise': (255, 255, 0)
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. WEBRTC TRANSFORMER CLASS ---
class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Standard processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                # Get Face ROI and Preprocess
                roi = cv2.resize(cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB), (48, 48))
                img_pixels = np.expand_dims(roi, axis=0).astype('float32') / 255.0
                
                # Prediction
                prediction = model.predict(img_pixels, verbose=0)
                label = emotion_labels[np.argmax(prediction)]
                color = color_map.get(label, (255, 255, 255))

                # Draw Bounding Box (Inverting for display if needed)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except:
                continue

        return img

# --- 3. UI LAYOUT ---
st.title("🧠 Live AI Emotion Detector")
st.write("Click 'Start' to begin real-time emotion recognition through your browser.")

webrtc_streamer(
    key="emotion-recognition",
    video_transformer_factory=EmotionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
