## Face Recognition System
<img width="130" alt="Screenshot 2025-04-14 at 2 00 32 PM" src="https://github.com/user-attachments/assets/75f4a665-e95a-49b0-8f46-a5d9a4b46910" />  
<img width="120" alt="Screenshot 2025-04-14 at 2 00 50 PM" src="https://github.com/user-attachments/assets/8c655a86-974a-4f3a-8435-e2258cf41a00" />

This project is a robust and modular **Face Recognition System** that supports multiple face detection and recognition backends, and integrates emotion, age, and gender analysis using deep learning models. including:

-  Haar Cascade (OpenCV)
-  DNN (OpenCV deep neural networks)
-  Dlib (HOG + CNN-based models)
-  DeepFace (for emotion analysis)
-  TensorFlow/Keras (for gender classification)

 **Detect faces** in real-time from:
- A webcam
- An IP camera
- A saved video file or image

**Recognize those faces** figure out who the person is by comparing with known faces stored in the system

**Analyze facial attributes:**
- **Emotions** (happy, sad, angry, etc.)
- **Estimated age**
- **Predicted gender**
  
---
<img width="400" alt="Screenshot 2025-04-14 at 2 14 55 PM" src="https://github.com/user-attachments/assets/252f07c7-42ea-4610-89bc-f47ac705f6e5" />



## 🧩 Modular Backend Support

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

  **📸 Features**

-  **Accurate Face Detection** with support for classic and modern models
-  **Robust Face Recognition** using traditional (LBPH) and deep learning approaches
-  **Emotion Analysis** with multi-class emotion classification
-  **Age Estimation** using deep CNN models
-  **Gender Prediction** with Keras-based classifier
-  **Real-time Video Support** for both live camera and pre-recorded video
-  **Freshest Frame Grabber**: Custom threaded class to always fetch the latest frame from camera feed
-  **Unique Face Extraction**:
-  Automatically filters duplicate face data
-  Ensures each face is recognized only once per instance
