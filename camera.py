import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


class Camera:

    def __init__(self):

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))


        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()


        # Create an align object
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return color_frame, aligned_depth_frame
    

    def process_frames(self, color_frame, aligned_depth_frame):

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        pointcloud = rs.pointcloud()
        points = pointcloud.calculate(aligned_depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(640*480,3)# xyz
        vertices_color = color_image.reshape(640*480,3)

        return color_image, depth_image, vertices, vertices_color
    

    def deproject_pixel(self, pixel_coords, aligned_depth_frame):

        depth = aligned_depth_frame.get_distance(pixel_coords[0], pixel_coords[1])
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, pixel_coords, depth)




