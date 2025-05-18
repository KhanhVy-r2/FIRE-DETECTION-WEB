import cv2

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Found working webcam at index {i}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Camera {i}", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        print(f"Camera index {i} not working.")
