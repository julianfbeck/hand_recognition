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
from _thread import start_new_thread

from system_action import trigger, volumeToPercent, combiTrigger, sayText, openProgram

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'model', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'label', 'label_map.pbtxt')

NUM_CLASSES = 11
MIN_THRESHOLD = 0.70

#Counts the objects [Counter, was it seen bevore]
objectCounter = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

#Time which the Object has to be seen
TRIGGER_TIME = 10

#when no object appered in this time, the counter will be reset
RESET_TIME = 5

#appearence in Percent to trigger action
PERCENT_TRIGGER = 0.6


#mute speach output
muteSound = 1

#time inbetween actions
PAUSE_TIME = 50
pause = 0

#take a picture wehen 1
screenshot = 0

#endscript
end = 0

#if 1 print text
argumentsMissing = 0
MISSING_TIME = 20

############################################################################################################################

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

##########################################################################################################

#plays sound
def startSoundThread(textToSay):
    global muteSound
    print(textToSay)
    #define Method wich will be started by the Thread
    if muteSound == 1:
        start_new_thread(sayText,(textToSay,))  #startThread

#set Time to pause action trigger
def setPauseTimer():
    global pause
    pause = PAUSE_TIME

#set variable to take a screenshot in the next frame
def takeScreenshot():
    global screenshot
    screenshot = 1

#mute the speach output
def mudeSound():
    global muteSound
    if muteSound == 1:
        startSoundThread("Sprachausgabe Deaktiviert")
        muteSound = 0
    else:
        muteSound = 1
        startSoundThread("Sprachausgabe Aktiviert")

#set fariable to force to end the script in the next frame
def endscript():
    global end
    end = 1

######################################## actions 
#exampels
#startSoundThread("Thumbs Up")              #play sound
#combiTrigger(0x5B, 0x44)                   #Windows +D
#volumeToPercent(20)                        #Volume to 20 Percent
#ctypes.windll.user32.LockWorkStation()     #Lock station
#trigger(0xAD)                              #mute system
#openProgramm('cmd')                           #run cmd
#end = 1                                    #stop programm and close window
#screenshot = 1                             #take a screenshot

def actionAtPeace0():
    setPauseTimer()
    volumeToPercent(0)
    startSoundThread("Lautstärke auf 0")

def actionAtPeace1():
    setPauseTimer()
    volumeToPercent(20)
    startSoundThread("Lautstärke auf 20")

def actionAtPeace2():
    setPauseTimer()
    volumeToPercent(40)
    startSoundThread("Lautstärke auf 40")

def actionAtPeace3():
    setPauseTimer()
    volumeToPercent(60)
    startSoundThread("Lautstärke auf 60")

def actionAtPeace4():
    setPauseTimer()
    volumeToPercent(80)
    startSoundThread("Lautstärke auf 80")

def actionAtPeace5():
    setPauseTimer()
    volumeToPercent(101)
    startSoundThread("Lautstärke auf 100")

def actionAtThumbsUp():
    setPauseTimer()
    startSoundThread("Mute Computer")
    trigger(0xAD)

def actionAtThumbsDown0():
    setPauseTimer()

def actionAtThumbsDown1():
    setPauseTimer()
    mudeSound()

def actionAtThumbsDown2():
    setPauseTimer()

def actionAtThumbsDown3():
    setPauseTimer()

def actionAtThumbsDown4():
    setPauseTimer()
    takeScreenshot()
    startSoundThread("Screenshot")

def actionAtThumbsDown5():
    setPauseTimer()
    startSoundThread("Schicht im Schacht")
    endscript()

def actionATOk():
    setPauseTimer()
    startSoundThread("Desktop wird angezeigt")
    combiTrigger(0x5B, 0x44)        #Windows+D

def actionAtStop0():
    setPauseTimer()
    startSoundThread("Explorer wird geöffnet")
    openProgram('Explorer')

def actionAtStop1():
    setPauseTimer()
    startSoundThread("Konsole wird geöffnet")
    openProgram('cmd')

def actionAtStop2():
    setPauseTimer()
    startSoundThread("firefox wird geöffnet")
    openProgram('firefox')

def actionAtStop3():
    setPauseTimer()
    startSoundThread("code minus insiders wird geöffnet")
    openProgram('code-insiders')

def actionAtStop4():
    setPauseTimer()
    startSoundThread("h s minus karlsruhe.de wird geöffnet")
    openProgram('firefox https://www.hs-karlsruhe.de')

def actionAtStop5():
    setPauseTimer()
    startSoundThread("webmail. h s minus karlsruhe.de wird geöffnet")
    openProgram('firefox https://webmail.hs-karlsruhe.de')

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
    global argumentsMissing
    actionlist = []
    for i in range(0,NUM_CLASSES+1):
        if objectCounter[i][0] > int(TRIGGER_TIME*PERCENT_TRIGGER):  
            actionlist.append(i)        
        objectCounter[i][0] = 0
    
    #try to find argumentvalue for thumbsDown and activate function
    if 5 in actionlist:             #thumbsDown
        actionlist.remove(5)
        if 1 in actionlist:         #zahl 2
            actionlist.remove(1)
            actionAtThumbsDown2()
        elif 4 in actionlist:       #zahl 3
            actionlist.remove(4)    
            actionAtThumbsDown3()
        elif 7 in actionlist:       #zahl 1
            actionlist.remove(7)
            actionAtThumbsDown1()
        elif 8 in actionlist:       #zahl 0
            actionlist.remove(8)
            actionAtThumbsDown0()
        elif 10 in actionlist:      #zahl 5
            actionlist.remove(10)
            actionAtThumbsDown5()
        elif 11 in actionlist:      #zahl 4
            actionlist.remove(11)
            actionAtThumbsDown4()
        else:
            print("Error, no arguments")
            argumentsMissing=MISSING_TIME

    #try to find argumentvalue for stop and activate function
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
            argumentsMissing=MISSING_TIME

    #try to find peace for stop and activate function
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
            argumentsMissing=MISSING_TIME
    
    #trigger functions for gesture without any arguments
    for i in actionlist:
        if i == 6:
            actionATOk()
        elif i == 9:
            actionAtThumbsUp()

#generate String which contains all detected Objects by name
def generateDetectionString():
    counter = 0
    string = ""
    for obj in objectCounter:
        if obj[0]>6:
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
    timerg = round(1/(time2-time1+0.00001),2)
    time1 = time2
    return str(timerg) + " FPS"

####################################################################################################################

detectioncounter = 0 #count frames in which object were detected
emptyframecounter = 0 #count empty frames

def detect_objects(image_np, sess, detection_graph):
    global detectioncounter
    global emptyframecounter
    global pause
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


    #comine results form cv to be processed later
    combined = list(map(lambda x, y, z: [float(x), float(y),z.tolist()], scores[0], classes[0], boxes[0]))
    bigger_than_threshold = list(filter(lambda x: x[0] >= MIN_THRESHOLD, combined))


    #when objects are recognized, it is counted which objects are there.
    # also detectioncounter is counted up. If this 10 was overruled, we got the object counter.
    #If no object is selected and it expires
    #the objectCounter and detectioncounter are set to 0.
    if pause > 0:   #pause detection
        pause -= 1
    else:
        if len(bigger_than_threshold) != 0 :
            countObjects(bigger_than_threshold)
            detectioncounter += 1
            emptyframecounter = 0
            if detectioncounter > TRIGGER_TIME:
                evaluateObjectcounter()
                detectioncounter = 0

        else:
            emptyframecounter+=1
            if emptyframecounter > RESET_TIME:
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
        min_score_thresh=MIN_THRESHOLD
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
        frame = cv2.flip(frame,1)
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
                #draw time lines with text
                cv2.line(   frame,(60,frame.shape[0]-5 ),(int((frame.shape[1]-60)*(emptyframecounter/RESET_TIME ))+60,frame.shape[0]-5 ),(0,0  ,255),10)
                cv2.line(   frame,(60,frame.shape[0]-15),(int((frame.shape[1]-60)*(detectioncounter/TRIGGER_TIME))+60,frame.shape[0]-15),(0,255,0  ),10)
                cv2.line(   frame,(60,frame.shape[0]-25),(int((frame.shape[1]-60)*(pause/PAUSE_TIME             ))+60,frame.shape[0]-25),(0,255,255),10)
                cv2.putText(frame,"resetTime"                        ,(2,frame.shape[0]-5 ), font, 0.3,(0,0  ,255),1,cv2.LINE_AA)
                cv2.putText(frame,"triggerTime"                        ,(2,frame.shape[0]-15), font, 0.3,(0,255,0  ),1,cv2.LINE_AA)
                cv2.putText(frame,"pauseTime"                        ,(2,frame.shape[0]-25), font, 0.3,(0,255,255),1,cv2.LINE_AA)
                cv2.putText(frame,generateDetectionString()             ,(0,frame.shape[0]-35), font, 1,(0,255,0),1,cv2.LINE_AA)
                #draw extra inforamtion
                cv2.putText(frame,("Sound on" if muteSound != 0 else "Sound off"),(5,25), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame, calcFPS()                            ,(frame.shape[1]-150,25), font, 1,(0,255,0),2,cv2.LINE_AA)
                #display "Missing Argument" for MISSING_TIME frames
                if argumentsMissing > 0:
                    argumentsMissing -= 1
                    cv2.putText(frame, "Missing Argument"                            ,(frame.shape[1]-250,frame.shape[0]-35), font, 1,(0,0,255),1,cv2.LINE_AA)
                cv2.imshow('Video', frame)
                if screenshot == 1:
                    cv2.imwrite('imgtest'+str(time.time())+'.jpg', frame)
                    screenshot = 0


        fps.update()

        font = cv2.FONT_HERSHEY_SIMPLEX
        

        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if (cv2.waitKey(1) & 0xFF == ord('q')) | end == 1:
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
