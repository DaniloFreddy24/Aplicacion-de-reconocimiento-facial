import streamlit as st
import cv2
import numpy as np
import requests
import os

st.set_page_config(page_title="Detector de Humanos", page_icon="👤", layout="centered")

st.title("🛡️ Sistema de Identificación Facial")
st.write("Ubícate frente a la cámara para iniciar el escaneo de verificación humana.")

MODEL_FILE = "haarcascade_frontalface_default.xml"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

@st.cache_resource
def get_detector():
    if not os.path.exists(MODEL_FILE):
        r = requests.get(MODEL_URL)
        with open(MODEL_FILE, "wb") as f:
            f.write(r.content)
    
    # Carga explícita del modelo
    detector = cv2.CascadeClassifier()
    detector.load(MODEL_FILE)
    return detector

try:
    face_cascade = get_detector()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

img_file_buffer = st.camera_input("📷 Presiona para escanear rostro")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) > 0:
        st.success("✅ **¡HUMANO IDENTIFICADO!** Verificación concedida.")
        st.balloons()
        for (x, y, w, h) in faces:
            cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.putText(cv2_img, 'HUMANO DETECTADO', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        st.error("⚠️ **ROSTRO NO HUMANO O NO DETECTADO.** No se reconoce una estructura facial humana válida.")
        st.warning("👉 Por favor, asegúrate de enfocar una cara humana con buena iluminación.")

    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_column_width=True)
