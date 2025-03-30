import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import serial
from tkinter import PhotoImage
import tkinter as tk
import time
from datetime import datetime
import webbrowser 


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


# Replace with your own phone number
recipient_phone_number = "0718684441"

# Create a tkinter window for code switching
root = tk.Tk()
root.title("Code Switcher")

# Create a frame to contain the buttons
# Create a frame
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)
# Create a canvas
canvas = tk.Canvas(frame, bg="lightblue", height=40)
canvas.pack(fill="both", expand=True)

# Add a headline label to the frame
headline_label = tk.Label( canvas,text="REMOTE PATIENT MONITORING", font=("Times New Roman", 18, "bold"), bg="lightblue")
headline_label.place(relx=0.5, rely=0.5, anchor="center")

root.configure(bg='white')

# Load an image file (replace 'path/to/your/image.png' with the actual path to your image file)
image_path = "/home/pi/Desktop/projectx/asd.png"
img = PhotoImage(file=image_path)

# Resize the image (adjust the factors according to your desired size)
resized_img = img.subsample(4, 4)  # Example: reduce the size by a factor of 2 in both dimensions

# Create a label to display the resized image
image_label = tk.Label(root, image=resized_img)
image_label.pack()

# Change the position of the image_label using the pack method
image_label.pack(side="bottom", padx=10, pady=10)  # Example: position at the bottom with padding




current_code = None  # Set the initial code to None

# Initialize the serial communication with SIM900C module
ser= serial.Serial("/dev/ttyS0", 9600, timeout=1)

# Function to send SMS using GSM SIM900A module
def send_sms_gsm(number, message):
    ser.write('AT+CMGF=1\r'.encode())
    time.sleep(1)
    ser.write('AT+CMGS="{}"\r'.format(number).encode())
    time.sleep(1)
    ser.write(message.encode())
    time.sleep(1)
    ser.write(chr(26).encode())  # Ctrl-Z to send the message
    


def open_predefined_website():
    website_url = "https://stem.ubidots.com/app/dashboards/647c2c8c3b8a71000e2fab26"
    webbrowser.open_new(website_url)


# Function to switch code
def switch_code():
    global current_code
    if current_code == code1:
        current_code = code2
        status_label.config(text="Sending Message Using Hand Sign")
    elif current_code == code2:
        current_code = code3
        status_label.config(text="Live Video")
    else:
        current_code = code1
        status_label.config(text="Patien Movement Alert")


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

def update_time():
    # Get the current time
    current_time = time.strftime("%H:%M %p")  # Format: HH:MM:SS AM/PM
    time_label.config(text=f"{current_time}")
    
    # Call this function again after 1000 milliseconds (1 second)
    root.after(1000, update_time)

def update_date():
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    date_label.config(text=f"{current_date}")

    # Call this function again after 86400000 milliseconds (24 hours)
    root.after(86400000, update_date)

# Add time and date labels
time_label = tk.Label(root, text="Time: ", font=("Helvetica",11,"bold"), bg="white")
time_label.pack(side="top", anchor="e", padx=10, pady=5)

date_label = tk.Label(root, text="Date: ", font=("Helvetica",11,"bold"), bg="white")
date_label.pack(side="top", anchor="e", padx=10, pady=5)

# Update time and date labels
update_time()
update_date()



# Create a "Switch Code" button with more customization
switch_button = tk.Button( text="Switch Option", command=switch_code, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
switch_button.pack(side="top", padx=10, pady=10)

# Create a "Run Code" button with more customization
run_button = tk.Button( text="Start Monitoring", command=run_current_code, bg="#3498db", fg="white", font=("Arial", 12, "bold"))
run_button.pack(side="top", padx=10, pady=10)

# Create a "Stop Code" button with more customization
stop_button = tk.Button( text="Stop Monitoring", command=stop_current_code, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"))
stop_button.pack(side="top", padx=10, pady=10)

open_predefined_website_button = tk.Button(
    text="Patient Health Readings", command=open_predefined_website, bg="#3498db", fg="white", border=0, font=("Microsoft YaHei UI Light", 12, "bold")
)
open_predefined_website_button.pack(side="top", padx=10, pady=10)



# Create a status label to show the current code
status_label = tk.Label(root, text="Current Option: None", font=("Helvetica", 12),bg="white")
status_label.pack(pady=10)

# Set the initial window size
root.geometry("450x580")

# Code 1
def code1():
    global prev_pred, message_sent, previous_landmarks, start_time,send_sms_gsm

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
                print("Landmarks not moving for 2 minutes")
                # Reset the timer to avoid repeated messages
                start_time = time.time()

            # Update previous landmarks for the next iteration
            previous_landmarks = lst

        if prev_pred is not None and prev_pred != pred:
            message_sent = False

        if prev_pred != pred and not message_sent:
            print(pred)
           # Send SMS using SIM900C
            send_sms_gsm("0718684441", f"Gesture Recognized: {pred}")
            prev_pred = pred
            message_sent = True

            

        # Display the video frame or perform any drawing as needed

        

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
        print(pred)
        # Send SMS using SIM900C
        send_sms_gsm("0718684441", f"Gesture Recognized: {pred}")
        prev_pred = pred
        message_sent = True

            

    # Display the video frame or perform any drawing as needed


def code3():
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        

# Release the camera and close the window
     cap.release()
     cv2.destroyAllWindows()

current_code = code1
status_label.config(text="Choose The Option")    

root.mainloop()
