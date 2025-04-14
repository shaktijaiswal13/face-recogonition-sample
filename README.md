#<img width="708" alt="Screenshot 2025-04-14 at 1 36 41 PM" src="https://github.com/user-attachments/assets/795c434b-0a4b-453b-b062-aba27d77b1b2" />
 Face Recognition System

This project is a robust and modular **Face Recognition System** that supports multiple face detection and recognition backends, including:

- ✅ Haar Cascade (OpenCV)
- ✅ DNN (OpenCV deep neural networks)
- ✅ Dlib (HOG + CNN-based models)
- ✅ DeepFace (for emotion analysis)
- ✅ TensorFlow/Keras (for gender classification)

---

## 📸 Features

- 👤 **Face Detection** using:
  - Haar Cascade
  - OpenCV DNN (`res10_300x300_ssd_iter_140000`)
  - Dlib face detector
- 🧠 **Face Recognition** using:
  - LBPH (Local Binary Pattern Histogram)
  - Face embeddings (Dlib, DeepFace)
- 🎭 **Emotion Detection** using DeepFace
- ⏳ **Age Estimation** using OpenCV DNN
- 🚻 **Gender Detection** using custom-trained Keras model
- 📹 **Real-time Video Support** via:<img width="704" alt="Screenshot 2025-04-14 at 1 36 58 PM" src="https://github.com/user-attachments/assets/dcd0514b-d687-4378-8626-5cb6377282c4" />

  - Live camera feed (USB/IP)
  - Pre-recorded video<img width="419" alt="Screenshot 2025-04-14 at 1 39 32 PM" src="https://github.com/user-attachments/assets/bb8cc75e-fb51-4a51-bad6-53385a8aff22" />

- 🧊 **Freshest Frame Grabber**: Custom threaded class to always fetch the latest frame from camera feed
- 🧬 **Unique Face Extraction** from multiple face captures using face embeddings
