import cv2
import mediapipe as mp
import numpy as np

# Inicializar el módulo de detección facial de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Crear ventanas para mostrar las diferentes imágenes
cv2.namedWindow('Face Mesh')
cv2.namedWindow('Contour')
cv2.namedWindow('Subtraction')
cv2.namedWindow('Thresholded')

# Inicializar el detector de puntos clave de la cara
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    
    # Convertir el frame de BGR a RGB (necesario para MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar puntos clave de la cara en el frame
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar puntos clave de la cara en la imagen original
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
            
            # Calcular el contorno de la cara
            mask = np.zeros_like(frame)
            mp_drawing.draw_landmarks(mask, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            contour = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, contour = cv2.threshold(contour, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                selected_contour = max(contours, key=cv2.contourArea)  # Seleccionar el contorno más grande
                cv2.drawContours(frame, [selected_contour], -1, (0, 0, 255), 1)  # Dibujar el contorno seleccionado
                
                # Mostrar la imagen del contorno
                cv2.imshow('Contour', frame)
                
                # Restar la imagen original del contorno
                subtraction = cv2.subtract(frame, mask)
                cv2.imshow('Subtraction', subtraction)
                
                # Umbralizar la imagen del contorno seleccionado
                _, thresholded = cv2.threshold(selected_contour, 1, 255, cv2.THRESH_BINARY)
                cv2.imshow('Thresholded', thresholded)
    
    # Mostrar el frame con los puntos dibujados
    cv2.imshow('Face Mesh', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
