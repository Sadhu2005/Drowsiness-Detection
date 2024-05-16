import face_recognition
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import pickle

# Function to perform speech recognition
def recognize_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Capture audio from microphone
    with sr.Microphone() as source:
        print("Please say your name:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Use recognizer to convert audio to text
        name = recognizer.recognize_google(audio)
        return name
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None

# Function to perform speech synthesis
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize lists to store known face encodings and names
try:
    with open("known_faces.pkl", "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
except FileNotFoundError:
    known_face_encodings = []
    known_face_names = []

# Initialize unknown face frames
unknown_face_frames = []

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video using HOG model
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # If face is unknown, add frame to unknown_face_frames list
            unknown_face_frames.append(frame)

            # If we have collected enough frames, train a new face recognition model
            if len(unknown_face_frames) >= 50:
                # Train face recognition model using frames
                # Implement your training logic here

                # Once trained, update known_face_encodings and known_face_names
                # known_face_encodings.append(new_face_encoding)
                # known_face_names.append(new_face_name)

                # Ask for the name of the unknown face
                speak("Please say your name.")
                name = recognize_speech()
                if name:
                    known_face_names.append(name)
                    known_face_encodings.append(face_encoding)
                    speak(f"Hello, {name}. Your face has been successfully recognized and added to the system.")
                    print(f"Hello, {name}. Your face has been successfully recognized and added to the system.")
                else:
                    speak("Name recognition failed. Please try again.")
                    print("Name recognition failed. Please try again.")

                # Clear unknown_face_frames list after training
                unknown_face_frames = []

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# Save known face encodings and names to file
with open("known_faces.pkl", "wb") as file:
    pickle.dump((known_face_encodings, known_face_names), file)
