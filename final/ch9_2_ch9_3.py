# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:03:16 2023

@author: BigData
"""

import numpy as np
import cv2 as cv
import sys

#%%

def construction_yolo_v3():
    f=open("coco_names.txt", 'r')
    # https://https://github.com/pjreddie/darknet/blob/master/data/coco.names
    class_names=[line.strip() for line in f.readlines()]
    
    model=cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img, 1./255., (448, 448), (0, 0,0), swapRB=True)
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    #print(output3); print(output3.shape)
    box, conf, id = [], [], []
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence > 0.5:
                centerx, centery = int(vec85[0]*width), int(vec85[1]*height)
                w, h = int(vec85[2]*width), int(vec85[3]*height)
                x, y = int(centerx-w/2), int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
                
    ind=cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

#%%Construct YOLO3

model, out_layers, class_names = construction_yolo_v3()
colors=np.random.uniform(0, 255, size=(len(class_names), 3))

#%% Check number of frames, elapsdes time to detect, fps
'''
import time
start=time.time()
n_frame=0
'''
#%% Input video and detect

#cap=cv.VideoCapture(0, cv.CAP_DSHOW)
cap=cv.VideoCapture("slow_traffic_small.mp4")
if not cap.isOpened():
    sys.exit("No camera or No Video")
    
while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit("fail to get frame")
    
    res=yolo_detect(frame, model, out_layers)
    


    for i in range(len(res)):
        x1, y1, x2, y2, confidence, id = res[i]
        text=str(class_names[id])+"%.3f"%confidence
        cv.rectangle(frame, (x1,y1),(x2,y2), colors[i])
        cv.putText(frame, text, (x1, y1+30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)
        
        cv.imshow("Object detection by YOLO3", frame)
        #n_frame+=1
    
    key=cv.waitKey(1)
    if key==ord('q'): break
#%%check elapsed time
'''



end=time.time()
print("Number of frames= ", n_frame, " Elaped time= ", end-start, "fps= ", n_frame/(end-start))
'''
#%%
cap.release()
cv.destroyAllWindows()   
