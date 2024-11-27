import cv2
print(cv2.getBuildInformation())

rtsp_url = "rtsp://admin:abcd@1113@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&resolution=64"  # office

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to connect to the camera stream.")
else:
    print("Connected to the camera stream successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Hikvision Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
