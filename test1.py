import cv2
from datetime import datetime


xml_haar_cascade = "haarcascade_frontalface_alt2.xml"
xml_haar_cascade2 = "haarcascade_mcs_nose.xml"


faceClassifier = cv2.CascadeClassifier(xml_haar_cascade)
noseClassifier = cv2.CascadeClassifier(xml_haar_cascade2)


# Camera

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame_color = capture.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 255, 0), 3)

        now = datetime.now()
        nose_rects = noseClassifier.detectMultiScale(gray, 1.5, 5)

        for (mx, my, mw, mh) in nose_rects:
            if y < my < y + h:
                cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(
                    frame_color, (mx, my), (mx + mw, my + mh), (255, 100, 0), 1
                )
                if nose_rects.all() > 0:
                    print(
                        now.strftime("[%d/%b/%Y %H:%M]")
                        + " Rosto potencialmente sem máscara detectado."
                    )
                break
    if nose_rects == ():
        print(now.strftime("[%d/%b/%Y %H:%M]") + " Nenhum risco potencial detectado")

    cv2.imshow("Mask detection", cv2.flip(frame_color, 1))
