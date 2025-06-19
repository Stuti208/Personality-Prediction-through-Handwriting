# ✍️ Personality Prediction through Handwriting

This project leverages deep learning techniques to **analyze handwriting samples** and predict personality traits using a **CNN-based model**. The system provides real-time predictions through a user-friendly web interface, enabling practical applications in psychology, career guidance, and behavioral assessment.

---

## 🧠 Project Overview

The core objective of this project is to explore the correlation between handwriting and personality traits using computer vision. A **Convolutional Neural Network (CNN)** was trained on labeled handwriting image data to identify patterns that correspond to predefined personality attributes.

The application includes a **Flask-based web interface** where users can upload handwriting images and instantly receive a personality analysis based on the trained model.

---

## 🔍 Key Features

- 🧠 **CNN Model for Personality Classification**
  - Built and trained a convolutional neural network for image-based classification.
  - Achieved reliable accuracy in predicting multiple personality traits.

- 🌐 **Real-time Web Interface**
  - Developed using **Flask**, allowing over 100 users to upload handwriting samples and receive instant predictions.
  - Simple UI for seamless interaction.

- 🖼️ **Image Preprocessing**
  - Automated image resizing, grayscale conversion, and normalization for model compatibility.

- 📊 **Scalable Deployment**
  - The architecture supports concurrent user interactions and can be scaled for broader applications.

---

## 🧱 Tech Stack

- **Programming Language**: Python  
- **Frameworks & Libraries**:
  - TensorFlow, Keras (Deep Learning)
  - Flask (Web Backend)
  - NumPy, OpenCV, PIL (Image Processing)
- **Frontend**: HTML, CSS

---

## ⚙️ How It Works

1. User uploads a handwriting image through the web UI.
2. Image is preprocessed and passed to the CNN model.
3. The model outputs predicted personality traits.
4. Predictions are displayed in real-time on the UI.
