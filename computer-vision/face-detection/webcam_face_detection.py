import numpy as np
import cv2

args = {
    'prototxt': 'deploy.prototxt.txt',
    'model': 'res10_300x300_ssd_iter_140000.caffemodel',
    'confidence': 0.6
}

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()
    (h, w) = img.shape[:2]
    frame = cv2.resize(img, (300, 300))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('camera', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
