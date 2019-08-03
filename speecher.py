import pyttsx3
import time
import numpy as np


def prune(res_steps):
    arr=res_steps
    i= 0
    count = 0
    max_co = 0
    unique_times = 0
    while i < len(arr):
        j =i+1
        count=0
        while j<len(arr) and abs(arr[i]-arr[j])<=5:
            j+=1
            count+=1
            if count > max_co:
                max_co = count
            if count >= 2:
                if count == 2:
                    unique_times+=1
                print(unique_times,"found",count,"\t",i,j,"\t", [arr[k] for
    k in range(i,j+1)])
        i = j
    print(max_co, unique_times)
    """ show_stat(y)
     y= np.delete(y, [168,169,170,171,172,173,224,225,226,227,228,229,230,  2
    46,247,248,249, 382,428, 882, 922, 939, 978, 979, 998, 1186, 1254, 1258,
    1259, 1267, 1386, 1444, 1458, 1497, 1712, 1869, 1934, 1971, 2008,2009, 2
    014, 2041, 2449, 2466, 2503, 2577 ])
    show_stat(y)
    y = np.delete(y, [2044,2045, 2199])
    """

engine = pyttsx3.init()
engine.setProperty('rate', 125)

x = np.load('res_steps_x2p.npy')

start=882
for i,n in enumerate(x[start:]):
    time.sleep(3.4)
    engine.say(str(n))
    engine.runAndWait()
    print(start+i, n)
    if (i % 60 == 0):
        engine.say("You did "+ str(i) +" this round!")
        engine.runAndWait()
    if (start+i % 1000 == 0):
        engine.say("Welcome to the "+str(start+i)+" millennial")
        engine.runAndWait()
