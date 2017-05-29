# interrupt-based distance measurement with median filtering
# remember to use GPIO.cleanup()

import RPi.GPIO as GPIO
import time

class ultrasonic():
    def __init__(self, echoPin, trigPin):
        # Pin definitions
        self.GND = 6 # only for reference
        self.ECHO = echoPin
        self.TRIG = trigPin
        self.VCC = 2 # only for reference

        # set up the board
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.TRIG,GPIO.OUT)
        GPIO.setup(self.ECHO,GPIO.IN)

        # let the sensor settle
        print "Waiting For Sensor To Settle"
        GPIO.output(self.TRIG, False)
        time.sleep(2)
        print "Sensor Settled"
        return
    
    def getDistance(self):
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)

        while GPIO.input(self.ECHO)==0:
            pulse_start = time.time()

        while GPIO.input(self.ECHO)==1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return distance
