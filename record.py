"""
Recording

@author: Morgen Malinoski
"""

import pyrealsense2 as rs
from skeletontracker import skeletontracker
import datetime
import json
import cv2
import numpy as np
import pickle
from depth_data import *
import os

# Record images, skeletons and depth data until keyboard interrupt is received, or exception occurs


def Record(saveDir, imageDir, pickleFile, depthPickle, depthIntrPickle, settingsFile, out_frame_data):
    try:
        # Configure depth and color streams of the intel realsense
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start the realsense pipeline
        pipeline = rs.pipeline()

        pipeline.start()

        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)

        # Get the intrinsics information for calculation of 3D point
        unaligned_frames = pipeline.wait_for_frames()
        frames = align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2

        # Create window for initialisation
        #window_name = "cuot workingbemos skeleton tracking with realsense D400 series"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

        # Write joint confidence and depth intrinsic
        settings = {"joint_confidence": joint_confidence}
        json.dump(settings, settingsFile)

        imageCounter = 0
        last_time = datetime.datetime.now()
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            unaligned_frames = pipeline.wait_for_frames()
            frames = align.process(unaligned_frames)
            depth = frames.get_depth_frame()

            #depth = DepthFramePickleable(frames.get_depth_frame())
            #depth.__getstate__ = DepthFrameGetState
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth.get_data())
            color_image = np.asanyarray(color.get_data())

            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(color_image)

            # Save skeletons along with frame counter
            pickle.dump((imageCounter, skeletons), pickleFile)

            # Create image
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Save image
            cv2.imwrite(os.path.join(
                imageDir, "image{}.png".format(imageCounter)), color_image)

            # Dump depth and depth intrinsic with frame counter
            pIntrinsics = IntrinsicsPickleable(depth_intrinsic)
            pickle.dump((imageCounter, pIntrinsics), depthIntrPickle)
            pickle.dump((imageCounter, DepthFramePickleable(
                depth, color_image)), depthPickle)

            # render the skeletons on top of the acquired image and display it
            '''cm.render_result(skeletons, color_image, joint_confidence)

            render_ids_3d(
                color_image, skeletons, depth, depth_intrinsic, joint_confidence
            )
            cv2.imshow(window_name, color_image)
            if cv2.waitKey(1) == 27:
                break'''

            # Get delta time in seconds
            now = datetime.datetime.now()
            delta = now - last_time
            delta = delta.total_seconds()
            last_time = now
            out_frame_data[imageCounter] = {"Delta Time": delta}

            imageCounter += 1

        pipeline.stop()
        # cv2.destroyAllWindows()

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
        pickleFile.close()
        depthPickle.close()
        depthIntrPickle.close()
