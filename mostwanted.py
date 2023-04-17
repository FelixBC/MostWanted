import cv2
import numpy as np
from keras.models import load_model

# Load the trained Keras model
model = load_model('keras_model.h5')

# Compile the model manually
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set the font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (0, 0, 255)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to a color image
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Resize the color image to the input shape of the model
    resized = cv2.resize(color, (224, 224))

    # Normalize the pixel values
    resized = resized / 255.0

    # Add a dimension to match the input shape of the model
    input_data = np.expand_dims(resized, axis=0)

    # Predict the face in the frame
    predictions = model.predict(input_data)

    # Get the predicted label
    predicted_label = np.argmax(predictions)

    # Set the label name based on the predicted label
    if predicted_label == 0:
        label_name = "Felix"
    elif predicted_label == 1:
        label_name = "Richard"
    elif predicted_label == 2:
        label_name = "Johan"
    else:
        label_name = "Unknown"

    # Draw a rectangle around the face
    if predicted_label != 3:
        font_color = (0, 0, 255)  # red
    else:
        font_color = (0, 255, 0)  # green

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), font_color, cv2.FILLED)
    cv2.putText(frame, label_name, (10, 35), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()