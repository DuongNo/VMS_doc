from fastapi import FastAPI,  UploadFile, File, Header, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
import os
from random import randint

import cv2
import time
import io
import PIL.Image as Image
import numpy as np
from typing import List
from numpy import random
from collections import Counter
import requests
import datetime
from facerecognition import faceRecogner

from typing import Union
from multiprocessing import Process, Value, Array 
import base64
import sys
from faceSystem import faceProcess
from traffic import Traffic
from app.config import Config
conf = Config.load_json('/home/vdc/project/computervision/python/VMS/app/config.json')
faceprocess_conf = Config.load_json('/home/vdc/project/computervision/python/VMS/faceprocess/faceSystem/config.json')
traffic_conf = Config.load_json('/home/vdc/project/computervision/python/VMS/traffic_process/traffic/config.json')

embeddings_path = faceprocess_conf.embeddings_path
faceReg = faceRecogner(faceprocess_conf, register=True)
 
processes = {"max":0}
streamers = {}

def file2image(file):
    image = Image.open(io.BytesIO(file)).convert('RGB') 
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

#######  Gstream HSL ####################################################

import queue
import gi
import logging

gi.require_version("GLib", "2.0")
gi.require_version("GObject", "2.0")
gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib, GObject

logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)8s] - %(message)s")
logger = logging.getLogger(__name__)

timestamp = 0
white = False
num_frames = 0

grabVec = queue.Queue()

def need_data(appsrc, length):
    global white, timestamp, grabVec, num_frames
    # print('need data...')
    size = 1080* 720 * 3
    buffer = Gst.Buffer.new_allocate(None, size, None)
    if grabVec.qsize() > 0:
        print(grabVec.qsize())
        buffer.fill(0, grabVec.get().tobytes())
        # print('Add frame...')
        buffer.duration = 1/30*Gst.SECOND
        buffer.pts = buffer.dts = int(num_frames * buffer.duration)
        buffer.offset = num_frames * buffer.duration
        num_frames += 1
    # else:
    #     Gst.Buffer.memset(buffer, 0, 0xff if white else 0x0, size)
    #     white = not white
    #     # print('need data...', white)
    #     # set buffer timestamp
    #     buffer.pts = timestamp
    #     buffer.duration = Gst.SECOND
    #     timestamp += buffer.duration

    # push buffer to appsrc
    retval = appsrc.emit('push-buffer', buffer)
    if retval != Gst.FlowReturn.OK:
        print("retval = ", retval)

def enough_data(src, size, length):
    print('Enough data')
    return Gst.FlowReturn.OK


def streamer(hlstrack_dir_loc, playlist_dir_loc, status):
    # sourcery skip: use-named-expression
    # Initialize GStreamer
    Gst.init(sys.argv[1:])
    
    # Create the elements
    appsrc = Gst.ElementFactory.make("appsrc", "appsrc")
    convert = Gst.ElementFactory.make("videoconvert", "convert")
    x264enc = Gst.ElementFactory.make("x264enc", "x264enc")
    mpegtsmux = Gst.ElementFactory.make("mpegtsmux", "mpegtsmux")
    hlssink = Gst.ElementFactory.make("hlssink", "hlssink")
    # sink = Gst.ElementFactory.make("appsink", "sink")

    # Create the empty pipeline
    pipeline = Gst.Pipeline.new("test-pipeline")

    if not pipeline or not appsrc or not convert or not x264enc or not mpegtsmux or not hlssink:
        logger.error("Not all elements could be created.\n")
        sys.exit(1)

    appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw, \
                                                    format=RGB, width=1080, \
                                                    height=720, framerate=30/1"))
    appsrc.set_property("format", "time")


    ## callback for need-data signal when appsrc out of data
    appsrc.connect("need-data", need_data)
    appsrc.connect('enough-data', enough_data)
    # x264enc.set_property('key-int-max', 25)
    hlssink.set_property("location", f"{hlstrack_dir_loc}/%05d.ts")
    hlssink.set_property("playlist-location", f"{playlist_dir_loc}/playlist.m3u8")
    hlssink.set_property("target-duration", 10)

    # Build the pipeline
    pipeline.add(appsrc)
    pipeline.add(convert)
    pipeline.add(x264enc)
    pipeline.add(mpegtsmux)
    pipeline.add(hlssink)

    if not appsrc.link(convert) or not convert.link(x264enc) or not x264enc.link(mpegtsmux) or not mpegtsmux.link(hlssink):
        logger.error("Elements could not be linked.")

        sys.exit(1)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Unable to set the pipeline to the playing state.\n")
        # gst_object_unref(data.pipeline);
        sys.exit(1)

    ## Bus pipeline
    bus = pipeline.get_bus()
    print('4.\n')

    terminate = False
    print("------------------ Init stream ----------------")
    print("------------------ Init stream ----------------status[0] = {}, status[1] = {}".format(status[0], status[1]))
    while not terminate and status[0] == 1:
        print("------------------ process image to stream ----------------first")
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        print("------------------ process image to stream ----------------two")
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug_info = msg.parse_error()
                logger.error(f"Error received from element {msg.src.get_name()}: {err.message}")
                logger.error(f"Debugging information: {debug_info or 'none'}")
                terminate = True
            elif msg.type == Gst.MessageType.EOS:
                logger.info("End-Of-Stream reached.")
                terminate = True
            elif msg.type == Gst.MessageType.STATE_CHANGED:
                if msg.source == pipeline:
                    old_state, new_state, pending_state = msg.parse_state_changed()
                    logger.error("Pipeline state changed from %s to %s:\n",
                            Gst.Element.state_get_name(old_state), Gst.Element.state_get_name(new_state))
            else:
                logger.error("Unexpected message received.")

    print('5.\n')
    pipeline.set_state(Gst.State.NULL)


##### End Gstream ######################################################

class AI_Process:
    def __init__(self, mode):
        self.conf = conf    
        self.mode = mode
        
    def initialize(self, type_camera, video_path):
        self.video_path = video_path
        self.face_processer = faceProcess(faceprocess_conf)
        self.traffic = None#Traffic(traffic_conf) 
        self.cap = cv2.VideoCapture(video_path)
        return
 
    def process(self, frame):
        ret = None
        if self.mode == "security":
            ret =  self.face_processer.facerecognize(frame)
            
        elif self.mode == "traffic":
            ret =  self.traffic.process(frame)
            
        return ret
    
    def image_process(self, status):
        frame_idx = 0 
        isSuccess = True    
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
        frame_idx = 6600#cong- 3500 #bach - 450, congBA-4800, 6600 - ca 3 nguoi
        #frame_idx = 35 #traffic
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while isSuccess and status[0] == 1: 
            isSuccess, frame = self.cap.read()
            start_time = time.time()
            if isSuccess:
                #print("frame_idx:",frame_idx)
                #print("frame.shape:",frame.shape)
                ret = self.process(frame)
                    
                end_time = time.time()
                time_process = end_time-start_time
                fps = 1/time_process
                fps = "FPS: " + str(int(fps))
                #print("frame_idx: {}  time process: {}  FPS: {}".format(frame_idx, time_process, fps))
                
                show_result = True
                save_video = False
                if show_result or save_video or status[1] == 1: # status.value == 2: send AI result to kafka server
                    if ret is not None:
                        for track in ret:
                            x1, y1, x2, y2 = track.bbox
                            track_id = track.track_id

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                            name = track.name
                            #if track.class_name is None:
                            #    name = track.name
                            #else:
                            #name = track.name + track.class_name + " " + track.lp
                            
                            txt = 'id:' + str(track.track_id) + "-" + name
                            org = (int(x1), int(y1)- 10)
                            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)

                if status[1] == 2:
                    frame = cv2.resize(frame, (1080, 720))
                    grabVec.put(frame)
                    print("------------------ put image to stream ----------------")
                if show_result:  
                    frame = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LINEAR)
                    cv2.imshow("test",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_idx +=1
                        
        print("end process with :",self.video_path)
        status[0] = 0
    
def initProcess(status, mode, type_camera, camera_path):
    print("Start process with status.value:",status[0])
    if mode == "0":
        mode = conf.default_security.mode
        type_camera = conf.default_security.type_camera
        camera_path = conf.default_security.camera_path
        print("running with default_security mode")
    elif mode == "1":
        mode = conf.default_traffic.mode
        type_camera = conf.default_traffic.type_camera
        camera_path = conf.default_traffic.camera_path
        print("running with default_traffic mode")
        
    processer = AI_Process(mode)
    processer.initialize(type_camera, camera_path)
    processer.image_process(status) 

WEB_SERVER = "http://172.16.50.91:8001/api/v1/attendance/attendance-daily"
app = FastAPI()

@app.get("/")
async def check():
    return {"message": "Hello World"}

@app.post("/init_process")
def init_process(
                        mode: str = Form(...),
                        type_camera: str = Form(...),
                        camera_path: str = Form(...),                     
                    ):  
    #-- 1 need module check connect to camera_path
    #-- 2 need mode for camera AI
    #mode: 0- defaul security, 1- default traffic, security,  traffic, 
    #type_camera: 1- camera, 2- cameraAI, 3- kafka
    #status(int): 0 - turn off process, 1 - run process, 2 - run process vs show results AI
    ret = 0
    message = "init process done"
    print("_______init process_______:")
    print("mode:",mode)
    print("type_camera:",type_camera)
    print("camera_path:",camera_path)
    
    '''
    if mode is None or type_camera is None:
        message = "mode, type_camera was None"
        return 1   
    if type_camera in ["camera", "cameraAI"]:
        if camera_path is None:
            message = "camera_path was None"
            return 1 
    '''
        
    status = Array('i', range(10))
    status[0] = 1
    status[1] = 0

    p = Process(name ="process-1", target=initProcess, args=(status, mode, type_camera, camera_path, ))
    p.start()

    processes["max"] +=1
    id_process = str(processes["max"])
    processes[id_process] = status
    
    hlstrack_dir_loc = '/home/vdc/project/computervision/python/VMS/public/hlstrack' # directory location of hlssink tracks
    playlist_dir_loc = '/home/vdc/project/computervision/python/VMS/public/playlist' # directory location of playlist streaming
    p2 = Process(name ="stream-1", target=streamer, args=(hlstrack_dir_loc, playlist_dir_loc, processes[id_process], ))
    p2.start()
         
    return {"result": ret,
            "message":message,
            "id_process": id_process}
    
@app.post("/adjust_process")
def adjust_process(                                    
                        id_process: str = Form(...),
                        status_value: int = Form(...)
                    ):  
    ret = 0
    message = "Adjust process done" 
    if id_process in processes.keys():    
        if status_value == 0:
            processes[id_process][0] = status_value
            processes.pop(id_process)
            
        elif status_value == 1:
            processes[id_process][1] = status_value
            
        elif status_value == 2:
            processes[id_process][1] = status_value
            
            '''
            hlstrack_dir_loc = '/home/vdc/project/computervision/python/VMS/public/hlstrack' # directory location of hlssink tracks
            playlist_dir_loc = '/home/vdc/project/computervision/python/VMS/public/playlist' # directory location of playlist streaming
            p = Process(name ="stream-1", target=streamer, args=(hlstrack_dir_loc, playlist_dir_loc, processes[id_process], ))
            p.start()
            '''
            
            
    else:
        message = "id_process {} were not exist".format(id_process)
        print(message)
        ret = -1
    
    return {"result": ret,
            "message":message}

@app.post("/updateinfo")
async def update_info(
                        member_names: List[str] = Form(...),
                        member_ids: List[int] = Form(...),
                        vectors: List[float] = Form(...)
                    ):
    ret = 0

    return {"result": ret}

@app.post("/register")
async def register(
                        file: UploadFile,
                        employee_name: str = Form(...),
                        employee_id: str = Form(...),
                        event: str = Form(...)
                    ):
    #3 event: 
    #1- register     - require: (employee_name, employee_id)
    #2- updateface   - require: (employee_name, employee_id)
    #3- remove       - require: (employee_id)

    print("employee_id:",employee_id)
    print("employee_name:",employee_name)
    
    vec = None
    result = 0
    if event == "register":
        contents = await file.read()
        image = file2image(contents)
        result, vec = faceReg.register(image, employee_name, employee_id, get_vector=True)
    elif event == "updateface":
        contents = await file.read()
        image = file2image(contents)
        result, vec = faceReg.updateFace(image, employee_name, employee_id, get_vector=True)
    elif event == "remove":
        result = faceReg.removeId(employee_id)
    elif event == "facebank":
        img = "/home/vdc/project/computervision/python/VMS/data/facedemo"
        faceReg.createFaceBank(img, embeddings_path)
        
    elif event == "evaluate":
        img = "/home/1.data/computervision/face/face_recognition/VN-celeb"
        #img = "/home/vdc/project/computervision/python/VMS/data/facedemo"
        embeddings_evaluate = "/home/vdc/project/computervision/python/VMS/data/evaluate/embedding"
        faceReg.createFaceBank(img, embeddings_evaluate)
        faceReg.evaluate(img, embeddings_evaluate)
        
    elif event == "clear":
        faceReg.clearInfo()
        faceReg.saveInfo()
        
    print("result:",result)
    return {"result": result,
            "employee_id": employee_id,
            "vector": vec}

    
@app.post("/get_status")
def get_status(
                        id_process: str = Form(...)
                    ): 
    ret = 0
    process_status = "OFF"
    
    if id_process in processes.keys():
        if processes[id_process][0] > 0:
            process_status = "ON"

    return {"result": ret,
            "process_status": process_status}

def updateInfoFromdatabase():
    BACKEND_SERVER = 'http://172.16.50.91:8001/api/v1/employer/all/' 
    headers =   {
                    'accept':"application/json"                 
                }
    res = requests.get(BACKEND_SERVER,json={"headers":headers})
    print("res:",res)
    print("type(res:",type(res))
    data = res["data"]
    info = []
    for d in data:
        full_name = d["full_name"]
        id = d["id"]
        face_vec = d["face_vector"]
        info.append([id, full_name, face_vec])
        print("full_name: {} _____ id: {}".format(full_name,id))

 
if __name__ == "__main__":
    video = "../video/face_video.mp4"
    mode = "security"
   

