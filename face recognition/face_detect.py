import cv2
import sys

# get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# create the haar cascade

face_cascade = cv2.CascadeClassifier(cascPath)

# read image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30))
        # flags = cv2.CV_HAAR_SCALE_IMAGE)
print('Found {0} faces'.format(len(faces)))

# draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('Faces Found', image)
cv2.waitKey(0)