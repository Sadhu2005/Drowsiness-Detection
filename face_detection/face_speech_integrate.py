import face_recognition
import cv2
import pyttsx3
import speech_recognition as sr

# Initialize speech recognition engine
recognizer = sr.Recognizer()

# Initialize speech synthesis engine
engine = pyttsx3.init()


# Function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        print("Please say your name:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        name = recognizer.recognize_google(audio)
        return name
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

# Initialize variables for face recognition
known_face_encodings = []
known_face_names = []

# Initialize font for putting text
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize variables for collecting frames for training
frames_to_collect = 30
frame_count = 0
frames = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            # Known face detected, get the name
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # Unknown face detected, ask for name
            engine.say("Please say your name.")
            engine.runAndWait()

            # Recognize speech and get the name
            name = recognize_speech()
            if name:
                # Collect frames for training
                frames.append(frame)
                frame_count += 1

                if frame_count >= frames_to_collect:
                    # Train face recognition model with collected frames
                    for collected_frame in frames:
                        collected_small_frame = cv2.resize(collected_frame, (0, 0), fx=0.25, fy=0.25)
                        collected_rgb_small_frame = collected_small_frame[:, :, ::-1]
                        face_locations = face_recognition.face_locations(collected_rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(collected_rgb_small_frame, face_locations)
                        for face_encoding in face_encodings:
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(name)

                    # Clear collected frames and reset frame count
                    frames = []
                    frame_count = 0

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name below the face
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
