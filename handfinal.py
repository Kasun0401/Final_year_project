import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from twilio.rest import Client

# Load the pre-trained Keras model and label data
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize the MediaPipe Holistic model
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Set up Twilio credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_phone_number'
your_phone_number = 'recipient_phone_number'

# Initialize the Twilio client
client = Client(account_sid, auth_token)

# Initialize the video capture from the default camera
cap = cv2.VideoCapture(0)

prev_pred = None
message_sent = False

while True:
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
        pred = label[np.argmax(model.predict(lst))]

    if prev_pred is not None and prev_pred != pred:
        message_sent = False

    if prev_pred != pred and not message_sent:
        print(pred)
        # Send a direct message using Twilio
        message = client.messages.create(
            body=f"Gesture Recognized: {pred}",
            from_=twilio_phone_number,
            to=your_phone_number
        )
        prev_pred = pred
        message_sent = True

    # Drawing code and displaying the video frame (as in your original script)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
