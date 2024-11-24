import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
tflite_model_path = "ktp_classifier2.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define image size
IMG_SIZE = (224, 224)

def preprocess_frame(frame):
    """Resize and normalize the frame."""
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

# Start webcam
cap = cv2.VideoCapture(1)  # Use 0 for the default webcam

print("Press 'q' to exit the webcam feed.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction result
    prediction = output_data[0][0]
    label = "Not KTP" if prediction > 0.5 else "KTP"

    # Display the result on the frame
    cv2.putText(frame, f"Prediction: {label} ({prediction:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
