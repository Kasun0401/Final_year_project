import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import vonage

# Load the pre-trained Keras model and label data
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize the MediaPipe Holistic model
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

client = vonage.Client(key="bf11ab8b", secret="SS0eGvNEsMbE77M0")
sms = vonage.Sms(client)

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
        # Send an SMS using Vonage (Nexmo)
        responseData = sms.send_message(
          {
        "from": "Vonage APIs",
        "to": "94718684441",
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
