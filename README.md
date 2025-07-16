🥔 Potato Disease Detection System
This project is an end-to-end deep learning solution designed to identify common potato leaf diseases (Early Blight, Late Blight, and Healthy) from images. It leverages a Convolutional Neural Network (CNN) for accurate classification and provides a robust FastAPI backend for real-time inference, coupled with a simple web-based user interface.

🌟 Features
Deep Learning Model: A custom Convolutional Neural Network (CNN) built with TensorFlow and Keras for image classification.

Disease Classification: Capable of identifying "Early Blight", "Late Blight", and "Healthy" potato leaves.

Robust Preprocessing: Includes image resizing, rescaling, and data augmentation techniques directly integrated into the model for improved generalization.

FastAPI Backend: A high-performance RESTful API for handling image uploads and returning real-time disease predictions.

Web User Interface: A simple HTML/CSS/JavaScript frontend for easy image selection, upload, and display of prediction results.

Efficient Model Loading: Model is loaded once at API startup using FastAPI's on_event handler for optimized performance.

🚀 Technologies Used
Backend:

Python

FastAPI

TensorFlow / Keras

NumPy

Pillow (PIL)

Uvicorn (ASGI server)

Frontend:

HTML5

CSS (Tailwind CSS via CDN)

JavaScript (Fetch API)
