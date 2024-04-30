import cv2
import numpy as np
from pynput.keyboard import Key, Controller

# Inicializar el controlador del teclado
keyboard = Controller()

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Voltear el fotograma horizontalmente
    frame = cv2.flip(frame, 1)
    
    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Umbralizar la imagen para obtener la máscara de la mano
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Encontrar los contornos en la máscara de la mano
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Buscar el contorno de la mano más grande
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calcular el área del contorno
        area = cv2.contourArea(max_contour)
        
        # Si el área es lo suficientemente grande, simular la pulsación de la tecla de espacio
        if area > 10000:
            print("Espacio detectado")
            keyboard.press(Key.space)
            keyboard.release(Key.space)
    
    # Mostrar la imagen
    cv2.imshow('Hand Detection', frame)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
