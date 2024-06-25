import cv2

# Substitua pelo endereço IP fornecido pelo IP Webcam, seguido de "/video"
url = "http://IP:PORT/video"

# Abra a câmera
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera do Celular', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
