import streamlit as st
import cv2
import numpy as np
import random
from PIL import Image, ImageOps, ImageDraw

st.set_page_config(page_title="Radar Cyberpunk - Verificación Humana", page_icon="🤖", layout="centered")

st.title("🤖 Radar Cyberpunk de Identificación Facial")
st.write("Demuestra tu autenticidad biológica ante el sistema.")

# Generar un reto aleatorio en cada sesión
if "challenge" not in st.session_state:
    challenges = [
        "Pon cara de sorprendido 😮 para verificar que no eres un androide",
        "Pica un ojo 😉 para la prueba de seguridad biológica",
        "Sonríe como si hubieras ganado la lotería 😁",
        "Pon cara seria de agente secreto 🕵️"
    ]
    st.session_state.challenge = random.choice(challenges)

st.info(f"🎯 **RETO DE AUTENTICIDAD:** {st.session_state.challenge}")

# Cargar el detector de rostros de OpenCV
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade()

img_file_buffer = st.camera_input("📷 Presiona para escanear rostro")

if img_file_buffer is not None:
    # Cargar y corregir la imagen
    image = Image.open(img_file_buffer)
    image = ImageOps.exif_transpose(image).convert("RGB")
    img_np = np.array(image)
    
    # Convertir a escala de grises para el detector
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    if len(faces) > 0:
        humanity_score = round(random.uniform(97.5, 99.9), 1)
        diagnostics = [
            "Sangre caliente detectada 🔥",
            "Pensamientos sobre pizza confirmados 🍕",
            "Sin signos de circuitos metálicos ⚡",
            "Pulso biológico dentro del rango normal ❤️"
        ]
        
        for (x, y, w, h) in faces:
            # Dibujar caja cian estilo Cyberpunk
            draw.rectangle([x, y, x + w, y + h], outline="#00FFCC", width=5)
        
        st.image(annotated_image, caption="Escaneo completado", use_container_width=True)
        st.success(f"✅ **¡HUMANO DETECTADO!** (Nivel de Humanidad: {humanity_score}%)")
        st.write(f"🔬 **Diagnóstico:** {random.choice(diagnostics)}")
        st.balloons()
    else:
        st.image(image, caption="Escaneo fallido", use_container_width=True)
        st.error("⚠️ **ROSTRO NO HUMANO O NO DETECTADO.** No se reconoce una estructura facial válida.")
        st.warning("👉 Enfoca tu cara, asegúrate de tener buena luz y cumplir con el reto.")

if st.button("🔄 Cambiar Reto"):
    del st.session_state.challenge
    st.rerun()
