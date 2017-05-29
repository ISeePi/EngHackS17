freerunningFinal.py
DETAILS
ACTIVITY
freerunningFinal.py
Sharing Info

General Info
Type
Text
Size
4 KB (3,705 bytes)
Storage used
0 bytesOwned by someone else
Location
All Python Scripts and Videos
Owner
Tin Vo
Modified
12:55 PM by Tin Vo
Opened
9:43 PM by me
Created
12:55 PM
Description
No description
Download permissions
Viewers can download


# import the necessary packages for picam
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# import for lane
import LaneNavigation as lnav
# import ultrasonic
from ultrasonic import ultrasonic

from subprocess import call

# initialize all
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
lastTime = 0
faceState = -1

# allow the camera to warmup
time.sleep(0.1)

ultrasonicInstance = ultrasonic(18,16)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        if x <= 0:
            faceState = -1
        elif x < 175:
            faceState = 2
        elif x >=175 and x<=450:
            faceState = 1
        elif x> 450:
            faceState = 0
        elif w*h>1800 and w*h < 4000:
            faceState = 3
        elif w*h >=4000 and w*h < 10000:
            faceState = 4
        elif w*h >=10000:
            faceState = 5

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWidows()

    sidewalkDirection = lnav.get_angle(image)
    distance = ultrasonicInstance.getDistance()
        
    currentTime = time.time()
    timeElapsed = currentTime - lastTime
    
    if timeElapsed > 7:
        lastTime = currentTime
        
        if distance < 50: # if about to collide
            print 'Ultrasonic: too close'
            call (["omxplayer", "/home/pi/Desktop/MP3s/STEP.mp3"])
        elif faceState == 0:
            print 'Face: right'
            call (["omxplayer", "/home/pi/Desktop/MP3s/RIGHT.mp3"])
        elif faceState == 1:
            print 'Face: centre'
            call (["omxplayer", "/home/pi/Desktop/MP3s/FRONT.mp3"])
        elif faceState == 2:
            print 'Face: left'
            call (["omxplayer", "/home/pi/Desktop/MP3s/LEFT.mp3"])
        elif faceState == 3:
            print 'Face: too close'
            call (["omxplayer", "/home/pi/Desktop/MP3s/FRONT.mp3"])
            call (["omxplayer", "/home/pi/Desktop/MP3s/STEP.mp3"])
        elif faceState == 4:
             print 'Face: mid-distance'
             call (["omxplayer", "/home/pi/Desktop/MP3s/FRONT.mp3"])
        elif faceState == 5:
            print 'Face: too close!'
            call (["omxplayer", "/home/pi/Desktop/MP3s/FRONT.mp3"])
            call (["omxplayer", "/home/pi/Desktop/MP3s/STEP.mp3"])
        elif sidewalkDirection == -1:
            # play audio for sidewalk going left
            print 'Lane: left'
            call (["omxplayer", "/home/pi/Desktop/MP3s/LEFT.mp3"])
        elif sidewalkDirection == 0:
            # play audio for sidewalk going straight
            print 'Lane: straight'
        elif sidewalkDirection == 1:
            # play audio for sidewalk going right
            print 'Lane: right'
            call (["omxplayer", "/home/pi/Desktop/MP3s/RIGHT.mp3"])
