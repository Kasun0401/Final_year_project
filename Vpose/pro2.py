import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import vonage
import tkinter as tk
import time

# Load the pre-trained Keras model and label data
model = load_model("model.h5")
label = np.load("labels.npy")

hmodel = load_model("hmodel.h5")
hlabel = np.load("hlabels.npy")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize the MediaPipe Holistic model
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize the Vonage client and SMS service
client = vonage.Client(key="c4af12f8", secret="BH7B19wepV6ZGp7x")
sms = vonage.Sms(client)

# Initialize the video capture from the default camera
cap = cv2.VideoCapture(0)

prev_pred = None
message_sent = False

# Set the threshold for movement
movement_threshold = 2  # Adjust based on your requirements
# Set the duration for inactivity
inactivity_duration = 10  # 2 minutes
# Initialize landmark positions
previous_landmarks = None
# Initialize timer
start_time = time.time()

# Create a tkinter window for code switching
root = tk.Tk()
root.title("Code Switcher")

# Create a frame to contain the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

current_code = None  # Set the initial code to None

# Create a text widget for displaying messages
text_widget = tk.Text(root, height=10, width=40, wrap="word")
text_widget.pack(pady=10)

# Function to switch code
def switch_code():
    global current_code
    if current_code == code1:
        current_code = code2
    else:
        current_code = code1

switch_button = tk.Button(button_frame, text="Switch Code", command=switch_code)
switch_button.pack(side="left", padx=10)

# Variable to track whether code should continue running
is_running = False

# Function to run the current code
def run_current_code():
    global is_running
    if current_code:
        is_running = True

        def run_code():
            if is_running:
                current_code()
                root.after(1, run_code)  # Call this function again in 1 millisecond

        run_code()

# Function to stop the code
def stop_current_code():
    global is_running
    is_running = False

# Function to clear the text widget
def clear_text_widget():
    text_widget.delete(1.0, tk.END)

# Create a "Stop Code" button
stop_button = tk.Button(button_frame, text="Stop Code", command=stop_current_code)
stop_button.pack(side="left", padx=10)

run_button = tk.Button(button_frame, text="Run Code", command=run_current_code)
run_button.pack(side="right", padx=10)

# Create a "Clear" button
clear_button = tk.Button(button_frame, text="Clear", command=clear_text_widget)
clear_button.pack(side="left", padx=10)

# Create a status label to show the current code
status_label = tk.Label(root, text="Current Code: None", font=("Helvetica", 12))
status_label.pack(pady=10)

# Set the initial window size
root.geometry("400x400")

# Code 1
def code1():
    global prev_pred, message_sent, previous_landmarks, start_time

    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    # Process the frame with Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = pose.process(frm_rgb)

        if results.pose_landmarks:
            for i in results.pose_landmarks.landmark:
                lst.append(i.x)
                lst.append(i.y)
                lst.append(i.z)  # Include z-coordinate as needed

        lst = np.array(lst).reshape(1, -1)

        # Check if there is no hand sign detected
        if np.count_nonzero(lst) == 0:
            pred = "patient is not in a bed"
        else:
            # Make a prediction using the pre-trained model
            pred = label[np.argmax(model.predict(lst))]

        # Check for movement
        if previous_landmarks is not None and previous_landmarks.size == lst.size:
            movement = np.linalg.norm(previous_landmarks - lst)
            if movement > movement_threshold:
                start_time = time.time()  # Reset the timer

            # Check if the specified duration has passed without movement
        if time.time() - start_time >= inactivity_duration:
                text_widget.insert(tk.END, "Landmarks not moving for 2 minutes\n")
                # Reset the timer to avoid repeated messages
                start_time = time.time()

            # Update previous landmarks for the next iteration
        previous_landmarks = lst

        if prev_pred is not None and prev_pred != pred:
            message_sent = False

        if prev_pred != pred and not message_sent:
            text_widget.insert(tk.END, f"Gesture Recognized: {pred}\n")
            # Send a direct message using Vonage
            responseData = sms.send_message(
                {
                    "from": "Vonage APIs",
                    "to": "94776949054",
                    'text': f"Gesture Recognized: {pred}"
                }
            )
            prev_pred = pred
            message_sent = True

        # Display the video frame or perform any drawing as needed

        status_label.config(text="Current Code: Code 1")

# Code 2
def code2():
    global prev_pred, message_sent

    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.left_hand_landmarks:
        for i in res.left_hand_landmarks.landmark:
            lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

    if res.right_hand_landmarks:
        for i in res.right_hand_landmarks.landmark:
            lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

    lst = np.array(lst).reshape(1, -1)

    # Check if there is no hand sign detected
    if np.count_nonzero(lst) == 0:
        pred = "normal"
    else:
        # Make a prediction using the pre-trained model
        pred = hlabel[np.argmax(hmodel.predict(lst))]

    if prev_pred is not None and prev_pred != pred:
        message_sent = False

    if prev_pred != pred and not message_sent:
        text_widget.insert(tk.END, f"Gesture Recognized: {pred}\n")
        # Send a direct message using Vonage
        responseData = sms.send_message(
            {
                "from": "Vonage APIs",
                "to": "94718684441",
                'text': f"Gesture Recognized: {pred}"
            }
        )
        prev_pred = pred
        message_sent = True

    # Display the video frame or perform any drawing as needed

    status_label.config(text="Current Code: Code 2")

# Set the initial code to Code 1
current_code = code1
status_label.config(text="Current Code: Code 1")

root.mainloop()
