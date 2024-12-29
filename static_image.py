import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained CNN model
model = load_model("asl_cnn.h5")  # Replace with your model file

# Define a function to preprocess the image
def preprocess_and_display_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Resize to 64x64 (or the size your model expects)
    resized_image = cv2.resize(image, (64, 64))
    # Normalize pixel values
    normalized_image = resized_image / 255.0
    # Reshape to match input shape (1, 64, 64, 3)
    reshaped_image = normalized_image.reshape(1, 64, 64, 3)
    
    # Display the image (convert back to displayable format)
    display_image = (reshaped_image[0] * 255).astype(np.uint8)  # Scale back to 0-255
    cv2.imshow("Preprocessed Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return reshaped_image

# Path to your test image
image_path = "tt1.png"  # Replace with your test image path

# Preprocess and display the image
processed_image = preprocess_and_display_image(image_path)

# Predict using the model
prediction = model.predict(processed_image, verbose=0)
predicted_label = np.argmax(prediction)
predicted_letter = chr(predicted_label + 65)  # Assuming class labels are A-Z

# Output the results
print(f"Predicted Letter: {predicted_letter}")
