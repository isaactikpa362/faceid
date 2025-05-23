import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Charger le classificateur en cascade pour la détection des visages
face_cascade = cv2.CascadeClassifier("face.xml")

st.title("🧠 Détection de Visages en Ligne")
st.write("Utilisez votre webcam pour capturer une image. Le système détectera les visages à l’aide de l’algorithme de Viola-Jones.")

# Choisir la couleur du rectangle
rect_color = st.color_picker("Choisissez la couleur du rectangle autour du visage", "#00FF00")

# Convertir couleur hex en tuple BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # OpenCV utilise BGR

# Paramètres ajustables
scaleFactor = st.slider("Paramètre scaleFactor", 1.05, 1.5, 1.1, 0.01)
minNeighbors = st.slider("Paramètre minNeighbors", 3, 10, 5)

# Capture de l’image avec la webcam
img_file = st.camera_input("📷 Capturez une image")

if img_file is not None:
    # Lire l'image
    img = Image.open(img_file)
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Dessiner les rectangles
    bgr_color = hex_to_bgr(rect_color)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), bgr_color, 2)

    st.image(img_np, channels="RGB", caption="Résultat avec visages détectés")

    if st.button("💾 Enregistrer l’image avec visages détectés"):
        output_filename = "face_detected_output.png"
        cv2.imwrite(output_filename, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        st.success(f"Image enregistrée sous le nom : {output_filename}")
