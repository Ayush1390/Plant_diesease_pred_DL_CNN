# Plant Disease Prediction Using CNN and Pytorch

## Overview
This project is a deep learning-based web application that predicts plant diseases from leaf images using a fine-tuned ResNet-18 model. The model classifies images into three categories: Healthy, Powdery, and Rust. The application is built using Flask and PyTorch and provides an interactive UI for users to upload images and receive predictions.

## Features
- Uses a fine-tuned **ResNet-18** model for disease classification
- **Flask web application** for easy image upload and prediction
- Supports **GPU acceleration** if available
- User-friendly UI with **real-time predictions**
- Outputs **classification label** (Healthy, Powdery, Rust)

## Technologies Used
- **Deep Learning**: PyTorch, ResNet-18 (pretrained model)
- **Web Framework**: Flask
- **Image Processing**: PIL, Torchvision
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
- **Deployment**: Local server (Flask)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plant-disease-prediction.git
   cd plant-disease-prediction
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open the application in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
- Open the web application.
- Upload an image of a plant leaf.
- The model processes the image and predicts if the plant is **Healthy, Powdery, or Rust**.
- The result is displayed on the screen with the uploaded image.

## Model Training
- **Pretrained Model**: ResNet-18 was used and modified for 3-class classification.
- **Training Data**: The dataset consists of plant leaf images.
- **Optimization**: The model was fine-tuned and evaluated for accuracy.

## Folder Structure
```
/plant-disease-prediction
│── /static/uploads       # Stores uploaded images
│── /templates            # HTML templates for Flask
│── /model                # Trained model files
│── app.py                # Flask application
│── requirements.txt      # Dependencies
│── README.md             # Project Documentation
```

