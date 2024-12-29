import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained CNN model
model = load_model("asl_cnn.h5")  # Replace with your model file

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(frame, (64, 64))      # Resize to 64x64
    normalized_frame = resized_frame / 255.0             # Normalize pixel values
    return normalized_frame.reshape(1, 64, 64, 3)        # Add batch and channel dimensions

# Initialize variables for building sentences
sentence = ""
prev_letter = ""
frame_count = 0

# Start the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the top-right portion of the frame as the ROI
    height, width, _ = frame.shape
    top_right = frame[0:height // 2, width // 2:width]

    # Preprocess the ROI
    processed_image = preprocess_frame(top_right)

    # Predict the letter
    prediction = model.predict(processed_image, verbose=0)
    predicted_letter = chr(np.argmax(prediction) + 65)  # Assuming A-Z are class labels

    # Add the letter to the sentence if it changes and after some frame stability
    if predicted_letter != prev_letter:
        frame_count = 0
        prev_letter = predicted_letter
    else:
        frame_count += 1

    if frame_count > 10:  # Add the letter only if stable for 10 frames
        sentence += predicted_letter
        frame_count = 0

    # Display the sentence and the ROI
    cv2.putText(frame, f"Predicted Letter: {predicted_letter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (width // 2, 0), (width, height // 2), (255, 0, 0), 2)  # Draw ROI box

    # Show the frames
    cv2.imshow("ASL Sentence Predictor", frame)
    cv2.imshow("ROI", top_right)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
