import cv2  # Import OpenCV lib
import numpy as np
from torch import hub
import torch
import pandas as pd
from ultralytics import YOLO


import cv2, queue, threading

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  def release(self):
    self.cap.release()


cap = VideoCapture('rtmp://192.168.137.223/live/001')
device = 'cuda:0'
# device = 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
model.to(device)



while(True):
    # Capture frame-by-frame
    frame = cap.read()
    
    frame = cv2.resize(frame,(1500,700))
    results = model(frame)
    for index,row in results.pandas().xyxy[0].iterrows():
        cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255,0,0), 2)
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()