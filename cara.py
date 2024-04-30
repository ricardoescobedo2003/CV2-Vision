import cv2
import mediapipe as mp

# Inicializar el módulo de detección facial de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Crear una ventana para mostrar solo las líneas dibujadas
cv2.namedWindow('Lines Only')

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
            # Dibujar puntos clave de la cara
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
            
            # Mostrar solo las líneas dibujadas en la ventana 'Lines Only'
            lines_only = frame.copy()
            lines_only.fill(0)  # Rellenar la imagen con negro para borrar la imagen original
            mp_drawing.draw_landmarks(lines_only, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))
            cv2.imshow('Lines Only', lines_only)
    
    # Mostrar el frame con los puntos dibujados
    cv2.imshow('Face Mesh', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
