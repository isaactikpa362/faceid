import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Détection de Visages", layout="centered")

# Titre de l'application
st.title("📸 Détection de Visage via Webcam")
st.markdown("""
Utilisez le bouton ci-dessous pour prendre une photo avec votre webcam.
L'application détectera automatiquement les visages présents dans l'image.
""")

# Chargement du classificateur Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paramètres utilisateur
st.sidebar.header("Paramètres de détection")
color = st.sidebar.color_picker("Choisir la couleur du rectangle", "#00FF00")
scaleFactor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.2, 0.1)
minNeighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, 1)

# Interface caméra
img_data = st.camera_input("📷 Cliquez ici pour capturer une image")

if img_data is not None:
    # Lire l'image à partir de l'objet BytesIO
    file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Dessiner des rectangles sur les visages
    r, g, b = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (b, g, r), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image avec visages détectés", use_column_width=True)

    if st.button("💾 Enregistrer l'image avec détection"):
        filename = "face_detected_output.jpg"
        cv2.imwrite(filename, image)
        st.success(f"Image enregistrée sous le nom : {filename}")
