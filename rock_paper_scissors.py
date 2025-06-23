import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model (update the path as necessary)
model_path = r"C:\Users\DELL\Downloads\my_model.keras"
model = tf.keras.models.load_model(model_path)

# Mapping labels back to gestures
labels = {0: 'rock', 1: 'paper', 2: 'scissors'}

# Load test data
try:
    X_test = np.load(r"C:\Users\DELL\Downloads\X_test.npy")
    y_test = np.load(r"C:\Users\DELL\Downloads\y_test.npy")

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")

except FileNotFoundError:
    print("Test data not found. Skipping test accuracy evaluation.")

# Plot training history
try:
    history = np.load(r"C:\Users\DELL\Downloads\history.npy", allow_pickle=True).item()
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Training History')
    plt.show()
except FileNotFoundError:
    print("Training history not found. Skipping plot.")

def predict_gesture(frame, model):
    """
    Predict the gesture for a given frame using the trained model.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    prediction = np.argmax(model.predict(resized))
    return labels[prediction]

def game_logic(user_move):
    """
    Determine the computer's move and the game result.
    """
    moves = ['rock', 'paper', 'scissors']
    computer_move = np.random.choice(moves)
    
    if user_move == computer_move:
        result = 'Draw'
    elif (user_move == 'rock' and computer_move == 'scissors') or \
         (user_move == 'paper' and computer_move == 'rock') or \
         (user_move == 'scissors' and computer_move == 'paper'):
        result = 'You Win'
    else:
        result = 'Computer Wins'
    
    return computer_move, result

def main():
    """
    Main function for webcam-based gesture recognition.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        user_move = predict_gesture(frame, model)

        # Generate computer move and result
        computer_move, result = game_logic(user_move)

        # Display results on the webcam feed
        cv2.putText(frame, f'Your Move: {user_move}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Computer: {computer_move}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Result: {result}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Rock Paper Scissors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()