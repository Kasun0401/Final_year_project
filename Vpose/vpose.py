import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import vonage

# Load the pre-trained Keras model and label data
model = load_model("model.h5")
label = np.load("labels.npy")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

''' Set the desired video resolution
desired_width = 1280  # Change this to your desired width
desired_height = 720  # Change this to your desired height'''

client = vonage.Client(key="c4af12f8", secret="BH7B19wepV6ZGp7x")
sms = vonage.Sms(client)

# Initialize the video capture from the default camera
cap = cv2.VideoCapture(0)

prev_pred = None
message_sent = False

'''# Set the video capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)'''

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
  
        if prev_pred is not None and prev_pred != pred:
              message_sent = False  

        if prev_pred != pred and not message_sent:    
            print(pred)
             # Send a direct message using Twilio
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