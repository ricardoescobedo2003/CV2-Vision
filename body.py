import cv2
import mediapipe as mp

# Inicializar el módulo de pose de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Crear una ventana para mostrar solo las líneas dibujadas
cv2.namedWindow('Lines Only')

# Inicializar el detector de pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    
    # Convertir el frame de BGR a RGB (necesario para MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar pose (puntos clave del cuerpo) en el frame
    results = pose_detector.process(rgb_frame)
    if results.pose_landmarks:
        # Dibujar puntos en todo el cuerpo
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
        
        # Mostrar solo las líneas dibujadas en la ventana 'Lines Only'
        lines_only = frame.copy()
        lines_only.fill(0)  # Rellenar la imagen con negro para borrar la imagen original
        mp_drawing.draw_landmarks(lines_only, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
        cv2.imshow('Lines Only', lines_only)
    
    # Mostrar el frame con los puntos dibujados
    cv2.imshow('Pose Detection', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
