import os
import cv2
import time
import argparse
import numpy as np
import subprocess as sp
import json
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, HLSVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

#Imports for Systemactions
import ctypes
from ctypes import wintypes
import win32com.client
from _thread import start_new_thread

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'model', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'label', 'label_map.pbtxt')

NUM_CLASSES = 11
MIN_THRESHOLD = 0.90

#Counts the objects [Counter, was it seen bevore]
objectCounter = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

#Time which the Object has to be seen
triggerTime = 10

#when no object appered in this time, the counter will be reset
resetTime = 5

#mute speach output
muteSound = 1

#how often the object have to appeare in the timeslot to trigger
MIN_APPERANCE_IN_TIMESLOT = 5

##############################################################Trigger stuff
speaker = win32com.client.Dispatch("SAPI.SpVoice")
shell = win32com.client.Dispatch("WScript.Shell")
user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
KEYEVENTF_SCANCODE    = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731
VK_TAB  = 0x09
VK_MENU = 0x12

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT, # nInputs
                             LPINPUT,       # pInputs
                             ctypes.c_int)  # cbSize

# Functions
def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

#trigger hexKeyCode key
def trigger(hexKeyCode):
    PressKey(hexKeyCode)
    ReleaseKey(hexKeyCode)

#change volume to percent%
def volumeToPercent(percent):
    for i in range (0, 100):
        trigger(0xAE)
    for i in range (0,(int)(percent/2)):
        trigger(0xAF)

#trigger hexKeyCode key combination
def combiTrigger(key1, key2):
    PressKey(key1)
    PressKey(key2)
    ReleaseKey(key2)
    ReleaseKey(key1)

############################################################################################################################

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

##########################################################################################################

def startSoundThread(textToSay):
    global muteSound
    print(textToSay)
    #define Method wich will be started by the Thread
    if muteSound == 1:
        def sayText(textToSay):
            speaker.Speak(textToSay)
            print(textToSay)
        start_new_thread(sayText,(textToSay,))  #startThread

def stopProgramm():
    # Clean up
    video.release()
    cv2.destroyAllWindows()
    exit()


######################################## actions 
#exampels
#startSoundThread("Thumbs Up")              #play sound
#combitrigger(0x5B, 0x44)                   #Windows +D
#volumeToPercent(20)                        #Volume to 20 Percent
#ctypes.windll.user32.LockWorkStation()     #Lock station
#trigger(0xAD)                              #mute system
#shell.Run('cmd')                           #run cmd
#stopProgramm()                             #stop programm and close window   

def actionAtPeace0():
    startSoundThread("Lautstärke auf 0")
    #volumeToPercent(0)

def actionAtPeace1():
    startSoundThread("Lautstärke auf 20")
    #volumeToPercent(20)

def actionAtPeace2():
    startSoundThread("Lautstärke auf 40")
    #volumeToPercent(40)

def actionAtPeace3():
    startSoundThread("Lautstärke auf 60")
    #volumeToPercent(60)

def actionAtPeace4():
    startSoundThread("Lautstärke auf 80")
    #volumeToPercent(80)

def actionAtPeace5():
    startSoundThread("Lautstärke auf 100")
    #volumeToPercent(101)

def actionAtThumbsUp():
    startSoundThread("Rechner sperren")
    #ctypes.windll.user32.LockWorkStation()

def actionAtThumbsDown():
    global muteSound
    if muteSound == 1:
        startSoundThread("Sprachausgabe Deaktiviert")
        muteSound = 0
    else:
        muteSound = 1
        startSoundThread("Sprachausgabe Aktiviert")

def actionATOk():
    startSoundThread("Desktop wird angezeigt")
    #combitrigger(0x5B, 0x44)        #Windows+D

def actionAtStop0():
    startSoundThread("Explorer wird geöffnet")
    #shell.Run('Explorer')

def actionAtStop1():
    startSoundThread("Konsole wird geöffnet")
    #shell.Run('cmd')

def actionAtStop2():
    startSoundThread("firefox wird geöffnet")
    #shell.Run('firefox')

def actionAtStop3():
    startSoundThread("code-inseiders wird geöffnet")
    #shell.Run('code-inseiders')

def actionAtStop4():
    startSoundThread("hs-karlsruhe.de wird geöffnet")
    #shell.Run('start https://www.hs-karlsruhe.de')

def actionAtStop5():
    startSoundThread("webmail.hs-karlsruhe.de wird geöffnet")
    #shell.Run('start https://webmail.hs-karlsruhe.de')

####################################################################################

#count Objects by frame. Each object will be only 
#count once also if it occures multible times in the frame
def countObjects(objectList):
    for i in range(0,NUM_CLASSES+1):
        objectCounter[i][1] = 0

    for obj in objectList:
        bevore = objectCounter[int(obj[1])][1]
        if bevore == 0:
            objectCounter[int(obj[1])][0] += 1
            objectCounter[int(obj[1])][1] = 1

#check if the object was more then 6 times in the frame in one timeslot
def evaluateObjectcounter():
    actionlist = []
    for i in range(0,NUM_CLASSES+1):
        if objectCounter[i][0] > MIN_APPERANCE_IN_TIMESLOT:  
            actionlist.append(i)        
        objectCounter[i][0] = 0
    
    if 2 in actionlist:             #Stop
        actionlist.remove(2)
        if 1 in actionlist:         #zahl 2
            actionlist.remove(1)
            actionAtStop2()
        elif 4 in actionlist:       #zahl 3
            actionlist.remove(4)    
            actionAtStop3()
        elif 7 in actionlist:       #zahl 1
            actionlist.remove(7)
            actionAtStop1()
        elif 8 in actionlist:       #zahl 0
            actionlist.remove(8)
            actionAtStop0()
        elif 10 in actionlist:      #zahl 5
            actionlist.remove(10)
            actionAtStop5()
        elif 11 in actionlist:      #zahl 4
            actionlist.remove(11)
            actionAtStop4()
        else:
            print("Error, no arguments")

    if 3 in actionlist:             #peace
        actionlist.remove(3)
        if 1 in actionlist:         #zahl 2
            actionlist.remove(1)
            actionAtPeace2()
        elif 4 in actionlist:       #zahl 3
            actionlist.remove(4)    
            actionAtPeace3()
        elif 7 in actionlist:       #zahl 1
            actionlist.remove(7)
            actionAtPeace1()
        elif 8 in actionlist:       #zahl 0
            actionlist.remove(8)
            actionAtPeace0()
        elif 10 in actionlist:      #zahl 5
            actionlist.remove(10)
            actionAtPeace5()
        elif 11 in actionlist:      #zahl 4
            actionlist.remove(11)
            actionAtPeace4()
        else:
            print("Error, no arguments")
    
    for i in actionlist:
        if i == 5:
            actionAtThumbsDown()
        elif i == 6:
            actionATOk()
        elif i == 9:
            actionAtThumbsUp()
        else:
            print("error _ Nonesens or Argument",i)

#generate String which contains all detected Objects by name
def generateDetectionString():
    counter = 0
    string = ""
    for obj in objectCounter:
        if obj[0]>7:
            if counter == 1:
                string += "2 "
            elif counter == 2:
                string += "Stop "
            elif counter == 3:
                string += "Peace "
            elif counter == 4:
                string += "3 "
            elif counter == 5:
                string += "ThumbsUp "
            elif counter == 6:
                string += "Ok "
            elif counter == 7:
                string += "1 "
            elif counter == 8:
                string += "0 "
            elif counter == 9:
                string += "ThumbsDown "
            elif counter == 10:
                string += "5 "
            elif counter == 11:
                string += "4 "
        counter +=1
    return string

time1 = time.process_time()#used to calculate fps

#calculate FPS
def calcFPS():
    global time1
    time2 = time.process_time()
    timerg = round(1/(time2-time1),2)
    time1 = time2
    return str(timerg) + " FPS"

####################################################################################################################

detectioncounter = 0 #count frames in which object were detected
emptyframecounter = 0 #count empty frames

def detect_objects(image_np, sess, detection_graph):
    global detectioncounter
    global emptyframecounter
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})


    combined = list(map(lambda x, y, z: [float(x), float(y),z.tolist()], scores[0], classes[0], boxes[0]))
    
    bigger_than_threshold = list(filter(lambda x: x[0] >= MIN_THRESHOLD, combined))


    #when objects are recognized, it is counted which objects are there.
    # also detectioncounter is counted up. If this 10 was overruled, we got the object counter.
    #If no object is selected and it expires
    #the objectCounter and detectioncounter are set to 0.
    if len(bigger_than_threshold) != 0 :
        countObjects(bigger_than_threshold)
        detectioncounter += 1
        emptyframecounter = 0
        if detectioncounter > triggerTime:
            evaluateObjectcounter()
            detectioncounter = 0

    else:
        emptyframecounter+=1
        if emptyframecounter > resetTime:
            for i in range(0,11):
                objectCounter[i][0] = 0
            detectioncounter = 0
            emptyframecounter = 0


    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strin', '--stream-input', dest="stream_in", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
    args = parser.parse_args()

    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    if (args.stream_in):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream_in).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    while True:
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                            0.3, (0, 0, 0), 1)
            if args.stream_out:
                print('Streaming elsewhere!')
            else:
                cv2.line(frame,(0,frame.shape[0]-10),(int(frame.shape[1]*(detectioncounter/triggerTime)),frame.shape[0]-10),(0,255,0),6)
                cv2.line(frame,(0,frame.shape[0]-3),(int(frame.shape[1]*(emptyframecounter/resetTime)),frame.shape[0]-3),(0,0,255),6)
                cv2.putText(frame,generateDetectionString(),(0,frame.shape[0]-20), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame, ("Loud" if muteSound != 0 else "Mute"),(5,25), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame, calcFPS() ,(frame.shape[1]-150,25), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow('Video', frame)

        fps.update()

        font = cv2.FONT_HERSHEY_SIMPLEX
        

        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
