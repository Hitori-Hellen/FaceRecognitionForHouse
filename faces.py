import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {"person_name": 1}
with open('labels.pickle', 'rb') as f:
    old_labels = pickle.load(f)
    labels = {v:k for k,v in old_labels.items()}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf >= 45 and conf <= 80:
            print(id)
            print(labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (0,255,0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        if id != None:
            f = open("return.txt", "w")
            f.write("Name:" + labels[id] + "\n" + "Auth:true" + "\n")
            f.close()
        else:
            f = open("return.txt", "w")
            f.write("Name:None" + "\n" + "Auth:false" + "\n")
            f.close()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()