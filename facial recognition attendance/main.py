import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

video_capture=cv2.VideoCapture()

stud1_img = face_recognition.load_image_file("photos/beyonce.JPG")
stud1_encoding = face_recognition.face_encodings(stud1_img)[0]

stud2_img = face_recognition.load_image_file("photos/jakie.jpg")
stud2_encoding = face_recognition.face_encodings(stud2_img)[0]

stud3_img = face_recognition.load_image_file("photos/rose.jpg")
stud3_encoding = face_recognition.face_encodings(stud3_img)[0]

stud4_img = face_recognition.load_image_file("photos/brad-pitt.jpg")
stud4_encoding = face_recognition.face_encodings(stud4_img)[0]

known_face_encoding= [
    stud1_encoding,
    stud2_encoding,
    stud3_encoding,
    stud4_encoding
]

known_face_names=[
    "beyonce",
    "jakie",
    "rose",
    "brad-pitt"
]

students= known_face_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s= True
now = datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date, '.csv', 'w+', newline='')
inwriter=csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25 )
    rgb_small_frame = small_frame[:, :, ::,1]  # Assuming BGR format, extract red channel
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            if True in matches:  # Check if any match is found
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    now = datetime.now()
                    current_time = now.strftime("%H-%M-%S")
                    inwriter.writerow([name, current_time])

    cv2.imshow("attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyALLWindows()
f.close()






















