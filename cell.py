import cv2
import math 

# Classes de objetos (neste caso, apenas Coca-Cola)
classNames = ["Coca-Cola"]


# Substitua pelo endereço IP fornecido pelo IP Webcam, seguido de "/video"
url = "http://192.168.0.236:4747/video"

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

#Carregar modelo
model = "C:\\Igor\\best.pt"

# Função para realizar a detecção de objetos em um frame
def detect_objects(frame, model):
    results = model(frame, stream=True)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]
            detections.append((label, confidence, (x1, y1, x2, y2)))
    return detections

# Função para exibir os resultados de detecção no frame
def draw_detections(frame, detections):
    for label, confidence, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, f'{label} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame
