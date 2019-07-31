import pyttsx3
import time
import numpy as np


engine = pyttsx3.init()
engine.setProperty('rate', 125)

x = np.load('res.npy')

start=1059
for i,n in enumerate(x[start:]):
    time.sleep(3.4)
    engine.say(str(n))
    engine.runAndWait()
    print(start+i, n)
    if (i % 60 == 0):
        engine.say("You did "+ str(i) +" this round!")
        engine.runAndWait()
    if (start+i == 2000):
        engine.say("Welcome to the 2000 millennial")
        engine.runAndWait()
