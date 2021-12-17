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
import os
import threading, queue
import time
import math
import depth_data, settings

NUM_THREADS = 16

def CreateDepthFramePickleable(d, depth, render_image, skeletons, joint_confidence):
    distances = {}
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 5
    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons)):
        skeleton_2D = skeletons[skeleton_index]
        joints_2D = skeleton_2D.joints
        for joint_index in range(len(joints_2D)):
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                low_bound_x = max(
                    0,
                    int(
                        joints_2D[joint_index].x -
                        math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_x = min(
                    cols - 1,
                    int(joints_2D[joint_index].x +
                        math.ceil(distance_kernel_size / 2)),
                )
                low_bound_y = max(
                    0,
                    int(
                        joints_2D[joint_index].y -
                        math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_y = min(
                    rows - 1,
                    int(joints_2D[joint_index].y +
                        math.ceil(distance_kernel_size / 2)),
                )
                for x in range(low_bound_x, upper_bound_x):
                    for y in range(low_bound_y, upper_bound_y):
                        distances[x,y] = depth.get_distance(x, y)
    
    d.distances = distances

def SaveFrames(in_queue, in_image_dir, in_skel_pickle, in_depth_pickle, in_depth_intr_pickle):
    while True:
        frame_number, data_frame = in_queue.get()

        if frame_number == "Quit":  # Quit signal
            in_queue.task_done()
            print("Thread quitting...")
            break

        # Save image
        color_image = cv2.cvtColor(data_frame.color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(
                in_image_dir, "image{}.png".format(frame_number)), color_image)
        
        # Convert intrinsics to pickleable object
        pIntrinsics = depth_data.IntrinsicsPickleable(data_frame.depth_intrinsic)
        #depth = depth_data.DepthFramePickleable(data_frame.depth, data_frame.color_image)
        
        # Save pickles
        pickle.dump((frame_number, pIntrinsics), in_depth_intr_pickle) # Depth intrinsic
        pickle.dump((frame_number, data_frame.depth), in_depth_pickle) # Depth
        pickle.dump((frame_number, data_frame.skeletons), in_skel_pickle) # Skeletons

        in_queue.task_done()

# Record images, skeletons and depth data until keyboard interrupt is received, or exception occurs


def Record(saveDir, imageDir, pickleFile, depthPickle, depthIntrPickle, settingsFile, out_frame_data):
    # Configure depth and color streams of the intel realsense
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, settings.frames_per_second)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, settings.frames_per_second)

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
    settings_dict = {"joint_confidence": joint_confidence}
    json.dump(settings_dict, settingsFile)
    
    # Create queue and thread for saving
    save_queue = queue.Queue()
    p = threading.Thread(target=SaveFrames, args=(save_queue, imageDir, pickleFile, depthPickle, depthIntrPickle))
    p.start()
    imageCounter = 0
    last_time = datetime.datetime.now()
    try:
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
            
            depth_pickleable = depth_data.DepthFramePickleable()
            CreateDepthFramePickleable(depth_pickleable, depth, color_image, skeletons,joint_confidence)

            # Save skeletons along with frame counter
            #pickle.dump((imageCounter, skeletons), pickleFile)

            # Create image
            #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Save image
            '''cv2.imwrite(os.path.join(
                imageDir, "image{}.png".format(imageCounter)), color_image)'''

            # Dump depth and depth intrinsic with frame counter
            #pIntrinsics = depth_data.IntrinsicsPickleable(depth_intrinsic)
            '''pickle.dump((imageCounter, pIntrinsics), depthIntrPickle)
            pickle.dump((imageCounter, depth_data.DepthFramePickleable(
                depth, color_image)), depthPickle)'''

            # render the skeletons on top of the acquired image and display it
            '''cm.render_result(skeletons, color_image, joint_confidence)

            render_ids_3d(
                color_image, skeletons, depth, depth_intrinsic, joint_confidence
            )
            cv2.imshow(window_name, color_image)
            if cv2.waitKey(1) == 27:
                break'''
                
            data_frame = depth_data.DataFrame(color_image, skeletons,
                depth_pickleable, depth_intrinsic)
            save_queue.put((imageCounter, data_frame))

            # Get delta time in seconds
            now = datetime.datetime.now()
            delta = now - last_time
            delta = delta.total_seconds()
            expected_delta = 1.0 / settings.frames_per_second
            if delta < expected_delta:
                time.sleep(expected_delta - delta)
                now = datetime.datetime.now()
                delta = now - last_time
                delta = delta.total_seconds()
            last_time = now
            out_frame_data[imageCounter] = {"Delta Time": delta}

            imageCounter += 1

        pipeline.stop()
        # cv2.destroyAllWindows()
    except KeyboardInterrupt:
        save_queue.put(("Quit", None))
        save_queue.join()
        p.join()
        print("Closing files...")
        pickleFile.close()
        depthPickle.close()
        depthIntrPickle.close()
