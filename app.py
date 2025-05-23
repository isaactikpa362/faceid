import cv2
import streamlit as st
import time

# Charge le classificateur de visage (modifie le chemin si besoin)
face_cascade = cv2.CascadeClassifier('face.xml')

def detect_faces(scaleFactor, minNeighbors, rect_color):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Impossible d'accéder à la webcam.")
        return

    stframe = st.empty()

    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la lecture de la webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors
        )

        # Convertir la couleur hex en BGR
        r = int(rect_color[1:3], 16)
        g = int(rect_color[3:5], 16)
        b = int(rect_color[5:7], 16)
        color_bgr = (b, g, r)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

        stframe.image(frame, channels="BGR")

        time.sleep(0.03)  # légère pause pour la fluidité

    cap.release()

def app():
    st.title("Détection de visages avec Viola-Jones")
    st.markdown("""
    **Instructions :**  
    - Cliquez sur **Démarrer la détection** pour activer la webcam et commencer la détection de visages.  
    - Cliquez sur **Arrêter la détection** pour stopper la webcam.  
    - Ajustez les paramètres **scaleFactor** et **minNeighbors** pour améliorer la détection.  
    - Choisissez la couleur des rectangles entourant les visages.  
    - Vous pouvez sauvegarder une image capturée avec les visages détectés.  
    """)

    scaleFactor = st.slider("scaleFactor", 1.01, 2.0, 1.1, 0.01)
    minNeighbors = st.slider("minNeighbors", 1, 10, 5)
    rect_color = st.color_picker("Couleur des rectangles", "#00FF00")

    if "detecting" not in st.session_state:
        st.session_state.detecting = False
    if "frame" not in st.session_state:
        st.session_state.frame = None

    if not st.session_state.detecting:
        if st.button("Démarrer la détection"):
            st.session_state.detecting = True
            detect_faces(scaleFactor, minNeighbors, rect_color)
    else:
        if st.button("Arrêter la détection"):
            st.session_state.detecting = False

    if st.session_state.frame is not None:
        if st.button("Sauvegarder l'image"):
            filename = f"face_detected_{int(time.time())}.png"
            cv2.imwrite(filename, st.session_state.frame)
            st.success(f"Image sauvegardée : {filename}")

if __name__ == "__main__":
    app()

