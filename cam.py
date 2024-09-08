import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
from datetime import datetime

# Load your pre-trained model
model = load_model('gender_detection.keras')

# Open webcam
webcam = cv2.VideoCapture(0)

# Class labels
gender_classes = ['man', 'woman']

# Initialize data storage
data_store = []

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    # Apply face detection
    faces, confidences = cv.detect_face(frame)

    # Loop through detected faces
    for face in faces:
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for the model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender and age
        (age, gender) = model.predict(face_crop)[0]

        # Determine the label for gender
        gender_label = gender_classes[int(np.round(gender))]
        age_label = int(age)

        # Check if the person is a senior citizen
        if age_label > 60:
            citizen_status = "Senior Citizen"
        else:
            citizen_status = "Not Senior Citizen"

        # Display the label and bounding box rectangle on the output frame
        label = f"{gender_label}, {age_label} ({citizen_status})"
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Store data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_store.append([age_label, gender_label, timestamp])

    # Display the output frame
    cv2.imshow("Age and Gender Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

# Save the data to an Excel or CSV file
df = pd.DataFrame(data_store, columns=["Age", "Gender", "Time of Visit"])
df.to_csv('age_gender_data.csv', index=False)
