import pickle
import pyttsx3
import keyboard
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import serial

# Initialize the speech recognition engine
word_to_output = {'I love you': 0, 'yes': 1, 'no': 2, 'hello': 3, 'thank you': 4}
r = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50) # Decrease rate by 50 (adjust the value as needed)

# Load the hand gesture recognition model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe hands detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Increase the sensitivity of the hand detection
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Create a serial connection with the Arduino
ser = serial.Serial('COM13', 9600)  # Replace 'COM3' with the name of the serial port on your computer

# Define a function to run the speech recognition in a separate thread
def speech_recognition_thread():
    with sr.Microphone() as source:
        print("Speak now...")
        while True:
            # Capture audio input continuously
            audio = r.listen(source)

            try:
                # Use the recognizer to convert speech to text
                text = r.recognize_google(audio)
                print("You said:", text)

                # Check if the recognized text matches any of the words to recognize
                if text in word_to_output:
                    output = word_to_output[text]
                    print("Output:", output)

                    # Send the output number to the Arduino over the serial port
                    ser.write(output.to_bytes(1, byteorder='little'))

                else:
                    print("Word not recognized.")
                    output = 10
                    ser.write(output.to_bytes(1, byteorder='little'))

            except sr.UnknownValueError:
                print("Could not understand audio.")
                output = 10
                ser.write(output.to_bytes(1, byteorder='little'))
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            # Sleep for a short duration to give other threads a chance to run
            time.sleep(0.1)

# Start the speech recognition thread
speech_thread = threading.Thread(target=speech_recognition_thread)
speech_thread.start()

# Define the labels for the recognized hand gestures
labels_dict = {0: 'I love you', 1: 'yes', 2: 'no', 3: 'Hello', 4: 'thank you'}

# Run the hand gesture recognition in the main thread
while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Pad the list with zeros if there are less than 84 coordinates
            data_aux = data_aux[:84] + [0] * (84 - len(data_aux))

            x1 =int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # Convert predicted gesture into speech
            engine.say(predicted_character)
            engine.runAndWait()

          
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the speech recognition thread
speech_thread.join()

# Release the resources
cap.release()
cv2.destroyAllWindows()