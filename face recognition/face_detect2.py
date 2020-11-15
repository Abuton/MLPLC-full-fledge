import cv2
import sys

# get user supplied values
# imagePath = sys.argv[1]
cascPath = sys.argv[1]
# create the haar cascade

face_cascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
	# capture frome-by-frame:
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces= face_cascade.detectMultiScale(gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(30,40))
		# flags=cv2.CV_HAAR_SCALE_IMAGE)

	# draw a rectangle around the faces
	for (x, y, w, h) in faces:
	    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

	 # display frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# to quit
video_capture.release()
cv2.destroyAllWindows()