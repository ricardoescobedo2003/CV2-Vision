import cv2
import mediapipe as mp
import pyautogui

# Inicializar el módulo de detección de manos de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar el detector de manos
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Variables para control de gestos
finger_count = 0
hand_open = False

while cap.isOpened():
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    
    # Convertir el frame de BGR a RGB (necesario para MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar manos en el frame
    results = hands_detector.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Dibujar puntos clave de la mano
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
            
            # Obtener la posición de la punta del dedo índice (dedo 8) y del dedo medio (dedo 12)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_tip_y = int(index_tip.y * frame.shape[0])
            middle_tip_y = int(middle_tip.y * frame.shape[0])
            
            # Detectar gestos
            if index_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
                finger_count += 1
            else:
                finger_count = 0
            
            if finger_count == 1:
                # Levantar un dedo: Subir el volumen
                pyautogui.press('volumeup')
            elif finger_count == 2:
                # Levantar dos dedos: Bajar el volumen
                pyautogui.press('volumedown')
            elif finger_count == 5:
                # Abrir la mano: Presionar la tecla de espacio
                if not hand_open:
                    hand_open = True
                    pyautogui.press('space')
                    print("Espacio detectado")
            elif middle_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y:
                # Levantar el dedo medio: Presionar Shift + N
                pyautogui.hotkey('shift', 'n')
    
    # Mostrar el frame con los puntos dibujados
    cv2.imshow('Hand Gesture Control', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
