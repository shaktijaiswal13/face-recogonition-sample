## Face Recognition System
<img width="170" alt="Screenshot 2025-04-14 at 14 34 03" src="https://github.com/user-attachments/assets/0b9a71e8-dd00-4a09-a5e4-b5b08ad62500" />
➜

<img width="100" alt="Screenshot 2025-04-14 at 2 00 32 PM" src="https://github.com/user-attachments/assets/75f4a665-e95a-49b0-8f46-a5d9a4b46910" />          <img width="100" alt="Screenshot 2025-04-14 at 2 00 50 PM" src="https://github.com/user-attachments/assets/8c655a86-974a-4f3a-8435-e2258cf41a00" />

This project is a robust and modular **Face Recognition System** that supports multiple face detection and recognition backends, including:

-  Haar Cascade (OpenCV)
-  DNN (OpenCV deep neural networks)
-  Dlib (HOG + CNN-based models)
-  DeepFace (for emotion analysis)
-  TensorFlow/Keras (for gender classification)

---
<img width="684" alt="Screenshot 2025-04-14 at 2 14 55 PM" src="https://github.com/user-attachments/assets/252f07c7-42ea-4610-89bc-f47ac705f6e5" />



## 📸 Features

-  **Face Detection** using:
  - Haar Cascade
  - OpenCV DNN (`res10_300x300_ssd_iter_140000`)
  - Dlib face detector
-  **Face Recognition** using:
  - LBPH (Local Binary Pattern Histogram)
  - Face embeddings (Dlib, DeepFace)
-  **Emotion Detection** using DeepFace
-  **Age Estimation** using OpenCV DNN
-  **Gender Detection** using custom-trained Keras model
-  **Real-time Video Support** via:
  - Live camera feed (USB/IP)
  - Pre-recorded video
-  **Freshest Frame Grabber**: Custom threaded class to always fetch the latest frame from camera feed
-  **Unique Face Extraction** from multiple face captures using face embeddings
