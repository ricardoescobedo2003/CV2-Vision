import cv2
import mediapipe as mp
import webbrowser
import pyautogui
import time

# Inicializar el módulo de detección de pose de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar el detector de pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Bandera para controlar el envío de mensaje
mensaje_enviado = False

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
        # Dibujar puntos clave del cuerpo
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
        
        # Contar el número de puntos clave detectados
        num_keypoints = len(results.pose_landmarks.landmark)
        
        # Si se detecta un cuerpo, abrir WhatsApp y enviar un mensaje
        if num_keypoints > 15 and not mensaje_enviado:
            # Abrir WhatsApp web en el navegador
            webbrowser.open("https://web.whatsapp.com/")
            time.sleep(10)  # Esperar unos segundos para que cargue la página
            
            # Simular el ingreso de texto y enviar el mensaje
            pyautogui.click(x=200, y=200)  # Hacer clic en la zona de escritura
            time.sleep(1)
            pyautogui.typewrite("Hola desde Python!")  # Escribir el mensaje
            time.sleep(1)
            pyautogui.press('enter')  # Enviar el mensaje
            
            mensaje_enviado = True  # Marcar que se envió el mensaje
    
    # Mostrar el frame con los puntos dibujados
    cv2.imshow('Pose Detection', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
