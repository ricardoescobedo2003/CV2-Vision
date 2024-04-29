import cv2

# Cargar el clasificador preentrenado para detectar rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para dibujar las líneas en el rostro
def draw_lines(img, faces):
    for (x, y, w, h) in faces:
        # Calcular puntos para las líneas
        start_point = (x, y + h//2)
        end_point = (x + w, y + h//2)
        color = (255, 0, 0)  # Color en formato BGR (azul, verde, rojo)
        thickness = 2  # Grosor de la línea
        # Dibujar la línea horizontal
        cv2.line(img, start_point, end_point, color, thickness)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame")
        break
    
    # Convertir el frame a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Dibujar líneas en los rostros detectados
    draw_lines(frame, faces)
    
    # Mostrar el frame con las líneas dibujadas
    cv2.imshow('Video', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
