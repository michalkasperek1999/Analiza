import boto3
import cv2
from botocore.exceptions import ClientError
import numpy as np
import time
STREAM_NAME = "PiStream"
kvs = boto3.client("kinesisvideo")
# Grab the endpoint from GetDataEndpoint
endpoint = kvs.get_data_endpoint(
    APIName="GET_HLS_STREAMING_SESSION_URL",
    StreamName=STREAM_NAME
)['DataEndpoint']
# Grab the HLS Stream URL from the endpoint
kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)
url = kvam.get_hls_streaming_session_url(
    StreamName=STREAM_NAME,
    PlaybackMode="LIVE"
)['HLSStreamingSessionURL']
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(url)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time= time.time()
frame_id = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(width), int(height)))

 
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
 
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
 
while True:
    success,img = cap.read()
    frame_id+=1
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
 
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
 
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
 
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(img,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),2)
    out2.write(img)
    try:
        url = kvam.get_hls_streaming_session_url(
        StreamName=STREAM_NAME,
        PlaybackMode="LIVE"
        )['HLSStreamingSessionURL']
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            break;
    
cap.release()    
out2.release()
cv2.destroyAllWindows()
txt = "output.avi"
s3 = boto3.client('s3')
timestr = time.strftime("%Y%m%d-%H%M%S")
with open(txt, "rb") as f:
    s3.upload_fileobj(f, "s3-raspi-1615-1999", "template/{}.avi".format(timestr))