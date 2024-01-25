import pyrealsense2 as rs
import numpy as np
import cv2
import time

from camera import Camera


if __name__ == '__main__':

    camera = Camera()
    time.sleep(1)

    count = 0

    while True:

        color_frame, aligned_depth_frame = camera.get_frames()

        color_image, depth_image, vertices, vertices_color = camera.process_frames(color_frame, aligned_depth_frame)

        camera_pose = np.random.rand(4,4) #TODO

        
        cv2.imwrite("PATH_TO_DATA_FOLDER/frame-%06d.color.jpg"%(count), color_image)
        cv2.imwrite("PATH_TO_DATA_FOLDER/frame-%06d.depth.png"%(count), depth_image)
        np.savetxt("PATH_TO_DATA_FOLDER/frame-%06d.pose.txt"%(count), camera_pose)


        count += 1
        time.sleep(1)







