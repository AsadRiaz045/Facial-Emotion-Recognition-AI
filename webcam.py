import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from collections import deque
import sys

# --- 1. Model Architecture & Weights ---
def build_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(48, 48, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Load weights (path handling)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_weights(os.path.join(base_dir, 'emotion_model.h5'))
    return model

model = build_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Color Mapping (Common for both)
color_map = {
    'Angry': (0, 0, 255), 'Happy': (0, 255, 0), 'Sad': (255, 0, 0),
    'Neutral': (200, 200, 200), 'Disgust': (0, 128, 0), 
    'Fear': (128, 0, 128), 'Surprise': (0, 255, 255)
}

# --- 2. Logic to detect if running in Streamlit ---
def is_streamlit():
    return 'streamlit' in sys.modules

# --- 3. EXECUTION ---
if is_streamlit():
    import streamlit as st
    st.title("🧠 AI Emotion Detector (Web Version)")
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=5)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while run:
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
    cap.release()

else:
    
    print("Running in Local OpenCV Mode... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    buffer = deque(maxlen=3)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = cv2.resize(cv2.cvtColor(gray[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB), (48, 48))
            img = np.expand_dims(roi, axis=0).astype('float32') / 255.0
            
            pred = model.predict(img, verbose=0)
            buffer.append(pred[0])
            label = emotion_labels[np.argmax(np.mean(buffer, axis=0))]
            color = color_map.get(label, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
