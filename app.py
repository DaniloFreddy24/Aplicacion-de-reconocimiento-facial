import streamlit as st
import numpy as np
import random
import urllib.request
import os
import onnxruntime as ort
from PIL import Image, ImageOps, ImageDraw

st.set_page_config(page_title="Radar Cyberpunk - Verificación Humana", page_icon="🤖", layout="centered")

st.title("🤖 Radar Cyberpunk de Identificación Facial")
st.write("Demuestra tu autenticidad biológica ante el sistema.")

# Generar reto aleatorio
if "challenge" not in st.session_state:
    challenges = [
        "Pon cara de sorprendido 😮 para verificar que no eres un androide",
        "Pica un ojo 😉 para la prueba de seguridad biológica",
        "Sonríe como si hubieras ganado la lotería 😁",
        "Pon cara seria de agente secreto 🕵️"
    ]
    st.session_state.challenge = random.choice(challenges)

st.info(f"🎯 **RETO DE AUTENTICIDAD:** {st.session_state.challenge}")

# Cargar Modelo ONNX liviano de Detección Facial
MODEL_FILE = "version-RFB-320.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"

@st.cache_resource
def load_onnx_model():
    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    return ort.InferenceSession(MODEL_FILE)

session = load_onnx_model()

def detect_faces(image, confidence_threshold=0.7):
    # Preprocesamiento para la red neuronal (320x240)
    img_resized = image.resize((320, 240))
    img_np = np.array(img_resized, dtype=np.float32)
    img_np = (img_np - 127.0) / 128.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    input_name = session.get_inputs()[0].name
    confidences, boxes = session.run(None, {input_name: img_np})
    
    confidences = confidences[0]
    boxes = boxes[0]
    
    width, height = image.size
    detected_boxes = []
    
    for i in range(boxes.shape[0]):
        conf = confidences[i][1]
        if conf > confidence_threshold:
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            detected_boxes.append((x1, y1, x2, y2))
            
    return detected_boxes

img_file_buffer = st.camera_input("📷 Presiona para escanear rostro")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image = ImageOps.exif_transpose(image).convert("RGB")
    
    # Realizar detección
    faces = detect_faces(image)
    
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
        
        for (x1, y1, x2, y2) in faces:
            draw.rectangle([x1, y1, x2, y2], outline="#00FFCC", width=5)
        
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
