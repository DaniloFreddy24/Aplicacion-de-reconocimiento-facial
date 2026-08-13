import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Detector de Humanos", page_icon="👤", layout="centered")

st.title("🛡️ Sistema de Identificación Facial")
st.write("Ubícate frente a la cámara para iniciar el escaneo de verificación humana.")

img_file_buffer = st.camera_input("📷 Presiona para escanear rostro")

if img_file_buffer is not None:
    # Cargar imagen con PIL (sin depender de cv2)
    image = Image.open(img_file_buffer)
    image = ImageOps.exif_transpose(image)
    
    st.image(image, caption="Captura realizada con éxito", use_column_width=True)
    st.success("✅ **¡CAPTURA RECIBIDA!** Verificación en proceso.")
    st.balloons()
