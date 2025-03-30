import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import vonage
import time

# Load the pre-trained Keras model and label data
model = load_model("model.h5")
label = np.load("labels.npy")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

client = vonage.Client(key="c4af12f8", secret="BH7B19wepV6ZGp7x")
sms = vonage.Sms(client)

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

while True:
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

    # Drawing code and displaying the video frame (as in your original script)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
