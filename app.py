import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps, ImageDraw

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
    image = ImageOps.exif_transpose(image).convert("RGB")
    img_array = np.array(image)
    
    # Detección de rostros
    results = face_detector.process(img_array)
    
    # Copia para dibujar
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = image.size
    
    if results.detections:
        num_faces = len(results.detections)
        
        # Dibujar recuadro sobre cada rostro detectado
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * width)
            ymin = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)
            
            # Dibujar rectángulo verde alrededor del rostro
            draw.rectangle([xmin, ymin, xmin + w, ymin + h], outline="lime", width=5)

        st.image(annotated_image, caption="Rostro(s) marcado(s)", use_container_width=True)
        st.success(f"✅ **¡HUMANO IDENTIFICADO!** Se detectó {num_faces} rostro(s) humano(s) válido(s).")
        st.balloons()
    else:
        st.image(image, caption="Sin rostro detectado", use_container_width=True)
        st.error("⚠️ **ROSTRO NO HUMANO O NO DETECTADO.** No se reconoce una estructura facial humana en la imagen.")
        st.warning("👉 Por favor, asegúrate de enfocar bien tu rostro y tener buena iluminación.")
