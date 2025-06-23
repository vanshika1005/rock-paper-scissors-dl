# rock-paper-scissors-dl
"An AI-powered Rock-Paper-Scissors game using a CNN for real-time hand gesture recognition via webcam. Trained in Colab, tested in VS Code, and built with TensorFlow and OpenCV."

This project implements a real-time Rock-Paper-Scissors game using hand gesture recognition through a webcam. The system uses a TensorFlow model which I trained on colab to classify hand gestures as rock, paper, or scissors, then determines the game outcome against a computer opponent.

Features:

Real-time gesture classification using a CNN model

Game logic implementation with computer opponent

Visual feedback displaying moves and results directly on webcam feed

Model evaluation with test accuracy metrics

Training history visualization 

Requirements:

Python 3.x

TensorFlow (>=2.0)

OpenCV

NumPy

Matplotlib (for training history visualization)

Run the main script:

bash
python rps_game.py
Position your hand in front of the webcam to make gestures:

Rock: Closed fist

Paper: Open palm

Scissors: Two fingers extended

The system will display:

Your detected gesture

Computer's random move

Game result

Press 'Q' to exit

File Structure:

text

project/

├── rock_paper_scissors.py     # Main game implementation

├── my_model.keras             # Trained gesture classification model

├── X_test.npy                 # Test images dataset

├── y_test.npy                 # Test labels dataset

└── history.npy                # Training history data
How It Works
Frame Capture: Webcam captures real-time video

Preprocessing:

Convert to grayscale

Resize to 64x64 pixels

Normalize pixel values

Gesture Prediction: Trained CNN model classifies hand gesture

Game Logic:

Computer randomly selects move

Results determined using Rock-Paper-Scissors rules:

Rock beats Scissors

Paper beats Rock

Scissors beats Paper

Display: Overlays game information on video feed

Customization

Model Training:

Update model architecture in training script

Retrain with new gesture data

Save updated model using model.save('my_model.keras')

Game Logic:

Modify game_logic() function in code

Add new gestures by updating labels dictionary

Interface:

Adjust text position/size in cv2.putText() calls

Modify display colors and formatting

Troubleshooting:

Webcam Issues: Ensure camera is connected and accessible

Model Loading Errors: Verify correct path to my_model.keras

Data Not Found: Place test data files in specified locations

Low Accuracy: Retrain model with more diverse gesture images
