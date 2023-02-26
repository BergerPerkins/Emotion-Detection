
import cv2
from deepface import DeepFace

# Load face detection model
face_cascade = cv2.CascadeClassifier(r"C:\Users\Berger\anaconda111\AI class\CV\OpenCV_Datasets\Day 5\haarcascade_frontalface_alt.xml")

# Open video stream
cap = cv2.VideoCapture(0)

# Loop over video frames
while True:
    # Read frame from video stream
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract face image
        face = frame[y:y+h, x:x+w]
        
        # Detect emotions in the face image
        result = DeepFace.analyze(face, actions=['emotion'])
        
        # Get the dominant emotion
        emotions = result[0]['dominant_emotion']
        
        # Display emotion label on the frame
        cv2.putText(frame, emotions, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close all windows
cap.release()
cv2.destroyAllWindows()