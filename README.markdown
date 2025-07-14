# Drowsiness Detection Project

## Overview
This project implements a drowsiness detection system that monitors a person's eyes and mouth movements to detect signs of drowsiness. The system uses a Convolutional Neural Network (CNN) model to analyze facial features and determine if the person is drowsy based on whether their eyes or mouth are open or closed. The project leverages OpenCV for real-time video processing and the trained CNN model is saved with a `.keras` extension.

## Features
- **Real-time Drowsiness Detection**: Detects drowsiness by analyzing eye and mouth movements in video feed.
- **CNN Model**: Utilizes a Convolutional Neural Network to classify drowsiness based on facial features.
- **OpenCV Integration**: Processes live video input for facial feature detection.
- **Model Storage**: The trained model is saved in `.keras` format for easy reuse and deployment.


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Chandu1613/Drowsiness-Detection
   cd drowsiness-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a webcam or a video input device connected for real-time detection.

## Usage
1. **Prepare the Model**: Ensure the trained CNN model (saved as a `.keras` file) is placed in the project directory.
2. Run the main script to start the drowsiness detection:
   ```bash
   python drowsiness_detection.py
   ```
3. The system will access your webcam, process the video feed, and display alerts if drowsiness is detected based on eye and mouth movements.

## How It Works
1. **Video Capture**: OpenCV captures real-time video from the webcam.
2. **Facial Feature Detection**: The system detects the eyes and mouth using OpenCV's pre-trained Haar cascades or other face detection methods.
3. **Drowsiness Analysis**: The CNN model processes the detected facial features to determine if the eyes or mouth indicate drowsiness (e.g., prolonged eye closure or yawning).
4. **Alert Generation**: If drowsiness is detected, the system triggers an alert (e.g., visual warning).

## Model Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Preprocessed images of the face, focusing on the eyes and mouth regions.
- **Output**: Binary classification (Drowsy or Not Drowsy)
- **File Format**: `.keras` (Keras model file)

## Training the Model
To retrain or modify the CNN model:
1. Collect a dataset of labeled images (drowsy and non-drowsy states).
2. Preprocess the images (e.g., resizing, normalizing) to focus on eyes and mouth.
3. Train the CNN model using TensorFlow/Keras.
4. Save the trained model as `model.keras`.

## Future Improvements
- Enhance model accuracy with a larger and more diverse dataset.
- Add support for audio alerts or integration with IoT devices.
- Optimize the CNN model for faster real-time performance.
- Implement additional facial landmarks for more robust detection.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate documentation.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any questions or suggestions, please open an issue on the GitHub repository or contact [your-email@example.com].