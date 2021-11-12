import cv2
from yolov5.detect_function import vehicle_detection

import numpy as np
import time
from tornado import httpclient
import base64
from PIL import Image
import io
http_client = httpclient.HTTPClient()
classes = ['motorcycle', 'car', 'bus', 'truck']
def get_image(id_camera):
    response = http_client.fetch("http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={}".format(id_camera))
    image = Image.open(io.BytesIO(response.body))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def count_number_per_class(clses):
    data = {
        "motorcycle": 0,
        "car": 0,
        "bus": 0,
        "truck": 0
    }

    for c in clses:
        cls = classes[int(c)]
        data[cls] += 1
    
    print(data)

#get frame from api to detect 
def api_frame_detect(frame):
    #load model
    detect_net = vehicle_detection('yolov5s6_09_07_2021.pt')


    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter('result.mp4', 
                            cv2.VideoWriter_fourcc(*'MP4V'),
                            30, size)

    start_time=time.time()
    boxes, confs, clses, img = detect_net.predict(frame)

    end_time=time.time()
    print("FPS: {} fps".format(1/(end_time-start_time)))

    cv2.imshow('frame', img)
    # cv2.imshow('frame', cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    # Write the frame into the
    # file 'filename.avi'
    result.write(img)

def image_detect(img_path):
    #load model
    detect_net = vehicle_detection('yolov5s6_09_07_2021.pt')

    frame = cv2.imread(img_path)
   
    boxes, confs, clses, img = detect_net.predict(frame)
    # cv2.imshow('frame', cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv2.imshow('frame', img)
    cv2.imwrite(img_path.split('.')[0]+'_result.jpg',img)
    # print(frame.shape)

def image_detect_img(frame):
    #load model
    detect_net = vehicle_detection('yolov5s6_09_07_2021.pt')

    boxes, confs, clses, img = detect_net.predict(frame)
    # print(boxes)
    # print(confs)
    # print(clses)
    count_number_per_class(clses)
    # cv2.imshow('frame', cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv2.imshow('frame', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('test.jpg',img)

def video_detect(video_path):
    #load model
    detect_net = vehicle_detection('yolov5s6_09_07_2021.pt')

    video = cv2.VideoCapture(video_path)
    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False): 
        print("Error reading video file")
    
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(video_path.split('.')[0]+'_result.mp4', 
                            cv2.VideoWriter_fourcc(*'MP4V'),
                            30, size)
   
    while True:
        ret, frame = video.read()
        if ret == True:

            start_time=time.time()
            boxes, confs, clses, img = detect_net.predict(frame)

            end_time=time.time()
            print("FPS: {} fps".format(1/(end_time-start_time)))

            cv2.imshow('frame', img)
            # cv2.imshow('frame', cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
            # Write the frame into the
            # file 'filename.avi'
            result.write(img)

            # Press q on keyboard 
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break


    video.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")
   
if __name__ == '__main__':
    
    # img_path='kien.jpg'
    # image_detect(img_path)

    id_camera = "5d8cd542766c880017188948"  # Thu Duc
    # id_camera = "5efd47ae942cda00169edf5c"
    img = get_image(id_camera)
    image_detect_img(img)

    # video_path = 'cam_14.mp4'
    # video_detect(video_path)

