import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Detector de Humanos", page_icon="👤", layout="centered")

st.title("🛡️ Sistema de Identificación Facial")
st.write("Ubícate frente a la cámara para iniciar el escaneo de verificación humana.")

# Inicializar detector facial de MediaPipe
@st.cache_resource
def load_face_detector():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

face_detector = load_face_detector()

img_file_buffer = st.camera_input("📷 Presiona para escanear rostro")

if img_file_buffer is not None:
    # Cargar imagen y corregir orientación
    image = Image.open(img_file_buffer)
    image = ImageOps.exif_transpose(image)
    img_array = np.array(image.convert('RGB'))
    
    # Procesar la imagen para detectar rostros
    results = face_detector.process(img_array)
    
    st.image(image, use_container_width=True)
    
    # Validar si se detectó al menos un rostro humano
    if results.detections:
        num_faces = len(results.detections)
        st.success(f"✅ **¡HUMANO IDENTIFICADO!** Se detectó {num_faces} rostro(s) humano(s) válido(s).")
        st.balloons()
    else:
        st.error("⚠️ **ROSTRO NO HUMANO O NO DETECTADO.** No se reconoce una estructura facial humana en la imagen.")
        st.warning("👉 Por favor, asegúrate de enfocar bien tu rostro y tener buena iluminación.")
