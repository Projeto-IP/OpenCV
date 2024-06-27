import cv2
from ultralytics import YOLO, solutions

url = "http://192.168.5.168:4747/video"
model = YOLO("C:\\Users\\User\\Downloads\\ChairDetect.pt")
cap = cv2.VideoCapture(url)
cap.set(3, 640)
cap.set(4, 480)

ClassNames = {0: 'EmptyChair', 1: 'FullChair'}
classes_to_count = [0, 1]  # person and car classes for count

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    classes_names=ClassNames,
    draw_tracks=False,
    reg_pts=[[0, 0], [800, 0], [800, 500],[0,500]]
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)

cap.release()
cv2.destroyAllWindows()
