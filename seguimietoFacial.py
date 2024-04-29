import cv2

# Cargar el clasificador preentrenado para detectar rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar el objeto para el seguimiento del rostro
tracker = cv2.TrackerMIL_create()

# Variables para almacenar las coordenadas del punto en la frente
forehead_x, forehead_y = 0, 0

# Función para dibujar un círculo rojo en la frente y líneas en forma de cruz
def draw_forehead_circle_and_cross(img, x, y, width, height, window_width, window_height):
    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)  # Dibujar un círculo rojo
    
    # Calcular las coordenadas de las líneas en forma de cruz
    cross_length = min(window_width, window_height) // 2
    x_start, y_start = window_width // 2 - cross_length, window_height // 2
    x_end, y_end = window_width // 2 + cross_length, window_height // 2
    
    # Dibujar líneas en forma de cruz
    cv2.line(img, (x_start, y_start), (x_end, y_start), (0, 255, 0), 2)  # Línea horizontal
    cv2.line(img, (window_width // 2, y_start - cross_length), (window_width // 2, y_end + cross_length), (0, 255, 0), 2)  # Línea vertical

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Capturar el primer frame para la detección de rostros
ret, frame = cap.read()

# Detectar el rostro en el primer frame
faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Inicializar el tracker con la región del rostro detectado
for (x, y, w, h) in faces:
    bbox = (x, y, w, h)
    tracker.init(frame, bbox)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame")
        break
    
    # Actualizar el tracker y obtener la nueva posición del rostro
    success, bbox = tracker.update(frame)
    
    if success:
        # Extraer las coordenadas del rectángulo del rostro
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Calcular la posición de la frente (en la mitad del rectángulo del rostro)
        forehead_x = x + w // 2
        forehead_y = y - h // 4
        
        # Obtener las dimensiones del frame
        window_width, window_height = frame.shape[1], frame.shape[0]
        
        # Dibujar un círculo en la frente y líneas en forma de cruz
        draw_forehead_circle_and_cross(frame, forehead_x, forehead_y, w, h, window_width, window_height)
        
        # Mostrar las coordenadas del punto en la pantalla
        cv2.putText(frame, f'Forehead: ({forehead_x}, {forehead_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Mostrar el frame con el círculo, las líneas y las coordenadas
    cv2.imshow('Face Tracker', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
