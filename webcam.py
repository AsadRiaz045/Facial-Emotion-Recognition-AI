import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# --- 1. MODEL LOADING (WITH CACHING) ---
@st.cache_resource
def load_emotion_model():
    # Architecture rebuild for compatibility
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(48, 48, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Load weights
    if os.path.exists('emotion_model.h5'):
        model.load_weights('emotion_model.h5')
    return model

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# BGR Colors for display
color_map = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 255, 128),  # Light Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Green
    'Neutral': (200, 200, 200), # Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 255)  # Yellow
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. WEBRTC VIDEO PROCESSING CLASS ---
class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror effect
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                # Preprocessing
                roi = cv2.resize(cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB), (48, 48))
                img_pixels = np.expand_dims(roi, axis=0).astype('float32') / 255.0
                
                # Prediction
                prediction = model.predict(img_pixels, verbose=0)
                label = emotion_labels[np.argmax(prediction)]
                color = color_map.get(label, (255, 255, 255))

                # Drawing
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except:
                continue

        return img

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="AI Emotion Detector", page_icon="🧠")
st.title("🧠 Live Facial Emotion Recognition")
st.write("This app uses MobileNetV2 to detect emotions in real-time.")

# STUN servers for stable connection
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
