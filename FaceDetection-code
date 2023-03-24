import cv2
import sys

imagePath = sys.argv[1]
cascPath = sys.argv[2]

faceCascade = cv2.CascadeClassifier(cascPath)

# read and convert image
image = cv2.imread("imagePath")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
print("Found {0} faces!".format(len(faces)))

# show face detections
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
