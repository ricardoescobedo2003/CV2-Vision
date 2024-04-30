import cv2
import mediapipe as mp
#Hace un marcarado sobre las manos y el rostro
# Inicializar el módulo de detección de caras de MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar el detector de caras y manos
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

while cap.isOpened():
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    
    # Convertir el frame de BGR a RGB (necesario para MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar caras en el frame
    results = face_detector.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Dibujar rectángulo alrededor de la cara
            cv2.putText(frame, f'Face ({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Detectar manos en el frame
    rgb_frame.flags.writeable = False  # Para mejorar la velocidad de procesamiento
    hands_results = hands_detector.process(rgb_frame)
    rgb_frame.flags.writeable = True
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Dibujar puntos en las puntas de los dedos
            for landmark in hand_landmarks.landmark:
                x_lm, y_lm = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(frame, (x_lm, y_lm), 5, (0, 255, 0), -1)  # Dibujar círculo en cada punto
            # Dibujar contorno de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=5))

    # Mostrar el frame con las líneas y puntos dibujados
    cv2.imshow('Face and Hand Detection', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
