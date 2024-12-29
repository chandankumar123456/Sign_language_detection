import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("asl_cnn.h5")

# Define the image size used during training
img_size = (64, 64)

# Load class labels (ensure they are in alphabetical order with 'del', 'nothing', and 'space')
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize an empty string to store the predicted word
predicted_word = ""

# Define a maximum length for the predicted word
max_word_length = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Define the region of interest (ROI) for predictions (top-right corner)
    x1, y1, x2, y2 = width - 150, 10, width - 10, 150
    roi = frame[y1:y2, x1:x2]

    # Draw a rectangle for the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Preprocess the ROI for prediction
    roi_resized = cv2.resize(roi, img_size)
    roi_normalized = roi_resized / 255.0  # Normalize pixel values to [0, 1]
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)  # Add batch dimension

    # Ensure the input has 3 channels
    if roi_reshaped.shape[-1] == 1:
        roi_reshaped = np.repeat(roi_reshaped, 3, axis=-1)

    # Check if the ROI has significant content
    if np.mean(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)) > 10:  # Threshold to determine if ROI is empty (adjust as needed)
        # Make a prediction
        prediction = model.predict(roi_reshaped)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display the prediction only if confidence is high enough
        if confidence > 0.8:
            if predicted_class == 'space':
                predicted_word += ' '
            elif predicted_class == 'del':
                predicted_word = predicted_word[:-1]  # Remove the last character
            elif predicted_class != 'nothing':
                predicted_word += predicted_class

            # Truncate the predicted word if it exceeds the maximum length
            if len(predicted_word) > max_word_length:
                predicted_word = predicted_word[:max_word_length]

    # Display the predicted word
    cv2.putText(frame, predicted_word, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
