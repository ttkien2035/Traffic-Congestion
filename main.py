import cv2
from yolov5.detect_function import vehicle_detection

import numpy as np
import time

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
    # print(size)

if __name__ == '__main__':
    
    img_path='kien.jpg'
    image_detect(img_path)

    # video_path = 'cam_14.mp4'
    # video_detect(video_path)

