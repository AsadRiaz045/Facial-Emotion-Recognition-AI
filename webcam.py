import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from collections import deque
import sys

# --- 1. MODEL ARCHITECTURE & WEIGHTS LOADING ---
def build_model():
    # MobileNetV2 architecture rebuild taake version mismatch error na aaye
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(48, 48, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Path handling
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, 'emotion_model.h5')
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model
    return None

# Emotion Labels aur 7 classes ke liye updated colors (BGR format)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
color_map = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 255, 128),  # Light Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Green
    'Neutral': (200, 200, 200), # Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 255)  # Yellow
}

# --- 2. RUNTIME DETECTION ---
def is_streamlit():
    return 'streamlit' in sys.modules or any('streamlit' in arg for arg in sys.argv)

# --- 3. MAIN EXECUTION ---
model = build_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
prediction_buffer = deque(maxlen=5)

if is_streamlit():
    import streamlit as st
    st.title("🧠 AI Emotion Detector (Web & Local Hybrid)")
    run = st.checkbox('🚀 Start Camera')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                roi = cv2.resize(cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB), (48, 48))
                img = np.expand_dims(roi, axis=0).astype('float32') / 255.0
                pred = model.predict(img, verbose=0)
                prediction_buffer.append(pred[0])
                label = emotion_labels[np.argmax(np.mean(prediction_buffer, axis=0))]
                color = color_map.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except: continue
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

else:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            try:
                roi = cv2.resize(cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB), (48, 48))
                img = np.expand_dims(roi, axis=0).astype('float32') / 255.0
                pred = model.predict(img, verbose=0)
                prediction_buffer.append(pred[0])
                label = emotion_labels[np.argmax(np.mean(prediction_buffer, axis=0))]
                color = color_map.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except: continue
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
