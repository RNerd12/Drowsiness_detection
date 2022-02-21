from typing import Annotated
import cv2 as cv
import numpy as np
import dlib
from imutils import face_utils
import pyttsx3
import threading

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

noface = 0
sleep = 0
drowsy = 0
active = 0
yawn_times = 0
status = ""
color = (0, 0, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up/(2.0*down)
    # print(round(ratio,2))
    if(ratio > 0.25):
        return 2
    elif(ratio > 0.16 and ratio <= 0.25):
        return 1
    else:
        return 0


def yawned(a, b, c, d, e, f, g, h):
    up = compute(b, h) + compute(c, g) + compute(d, f)
    down = compute(a, e)
    ratio = up/(2.0*down)
    # print(round(ratio,2))
    if(ratio > 0.5):
        return 1
    else:
        return 0


alarm_sound = pyttsx3.init()
voices = alarm_sound.getProperty('voices')
alarm_sound.setProperty('voice', voices[0].id)
alarm_sound.setProperty('rate', 130)


def voice_alarm(alarm_sound):
    alarm_sound.say(val)
    alarm_sound.runAndWait()


ANCHOR_POINT = (320, 240)


def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point
    if ny > y + multiple * h:
        return True
    return '-'


(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_frame = frame.copy()
    faces = detector(gray)
    alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))
    if(len(faces) == 0):
        noface += 1
        if(noface > 10):
            status = "No face Detected"
            color = (255, 255, 255)
            val = "No face detected"
            alarm.start()
    elif(len(faces) == 1):
        noface = 0
    cv.putText(face_frame, status, (100, 100),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        yawn = yawned(landmarks[60], landmarks[61], landmarks[62], landmarks[63],
                      landmarks[64], landmarks[65], landmarks[66], landmarks[67])
        nose = landmarks[nStart:nEnd]
        nose_point = (nose[3, 0], nose[3, 1])
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 20, 70
        multiple = 1
        cv.rectangle(face_frame, (x - w, y - h), (x + w, y + h), RED_COLOR, 2)
        cv.line(face_frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)
        dir = direction(nose_point, ANCHOR_POINT, w, h)

        if(left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if(sleep > 6):
                status = "SLEEPING"
                color = (255, 0, 0)
                val = "Please Wake up"
                alarm.start()

        if(dir == True):
            status = "Slouching warning"
            color = (0, 0, 255)
            val = "Slouching warning"
            alarm.start()

        elif(yawn == 1):
            yawn_times += 1
            print(yawn_times)
            if(yawn_times > 15):
                status = "Yawn Times Exceeded"
                color = (0, 0, 255)
                val = "Please pull over and take a break"
                alarm.start()

        elif(left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if(drowsy > 6):
                status = "Drowsy"
                color = (0, 0, 255)
                val = "drowsiness warning"
                alarm.start()

        elif((left_blink == 2 or right_blink == 2) and yawn_times < 15):
            drowsy = 0
            sleep = 0
            active += 1
            if(active > 6):
                status = "Active"
                color = (0, 255, 0)
                alarm_sound.stop()

        cv.putText(frame, status, (100, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # cv.imshow("Frame", frame)
    cv.imshow("Result of detector", face_frame)
    if cv.waitKey(200) & 0xFF == ord('d'):
        break
