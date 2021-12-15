#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
import argparse
import pickle
import os
import shutil
import zipfile, json
from skeletontracker import skeletontracker
from depth_data import *

def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence
):
    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 5
    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        did_once = False
        for joint_index in range(len(joints_2D)):
            if did_once == False:
                cv2.putText(
                    render_image,
                    "id: " + str(skeleton_2D.id),
                    (int(joints_2D[joint_index].x), int(
                        joints_2D[joint_index].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    thickness,
                )
                did_once = True
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                distance_in_kernel = []
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
                        distance_in_kernel.append(depth_map.get_distance(x, y))
                median_distance = np.percentile(
                    np.array(distance_in_kernel), 50)
                depth_pixel = [
                    int(joints_2D[joint_index].x),
                    int(joints_2D[joint_index].y),
                ]
                if median_distance >= 0.3:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic, depth_pixel, median_distance
                    )
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    point_str = [str(x) for x in point_3d]
                    cv2.putText(
                        render_image,
                        str(point_3d),
                        (int(joints_2D[joint_index].x),
                         int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )


def Record(saveDir, imageDir):
    # Create save files
    pickleFileLoc = os.path.join(saveDir, "save.skel")
    pickleFile = open(pickleFileLoc, "wb")
    #jcFileLoc = os.path.join(saveDir, "joint_confidence.sav")
    #jcPickle = open(jcFileLoc, "wb")
    depthFileLoc = os.path.join(saveDir, "depth.sav")
    depthPickle = open(depthFileLoc, "wb")
    depthIntrFileLoc = os.path.join(saveDir, "depth_intrinsic.sav")
    depthIntrPickle = open(depthIntrFileLoc, "wb")
    settingsFileLoc = os.path.join(saveDir, "settings.json")
    
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
        window_name = "cubemos skeleton tracking with realsense D400 series"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        
        # Write joint confidence and depth intrinsic
        settings = {"joint_confidence" : joint_confidence}
        with open(settingsFileLoc, 'w') as jf:
            json.dump(settings, jf)
        #with open(depthIntrFileLoc, 'w') as df:
            #pickle.dump(depth_intrinsic, df)

        imageCounter = 0
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
            pickle.dump((imageCounter, DepthFramePickleable(depth, color_image)), depthPickle)
            pickle.dump((imageCounter, IntrinsicsPickleable(depth_intrinsic)), depthIntrPickle)

            # render the skeletons on top of the acquired image and display it
            '''cm.render_result(skeletons, color_image, joint_confidence)

            render_ids_3d(
                color_image, skeletons, depth, depth_intrinsic, joint_confidence
            )
            cv2.imshow(window_name, color_image)
            if cv2.waitKey(1) == 27:
                break'''
            
            imageCounter += 1

        pipeline.stop()
        # cv2.destroyAllWindows()

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
        pickleFile.close()
        depthPickle.close()
        depthIntrPickle.close()

def IsDirEmpty(dir):
    if not os.path.isdir(dir):
        return True
    
    return len(os.listdir(dir)) == 0


def GetValidNewDirFromFilename(fullPath):
    basename = os.path.basename(fullPath)
    if basename[-4:] == ".zip":  # Check if it has the zip file extension
        basename = basename[:-4]
    else:
        basename = basename + "_extract"
        
    # String digits to the end. Not very clean though.
    sub = 0
    while os.path.isdir(os.path.join(os.path.dirname(fullPath), basename)):
        basename += "{}".format(sub)
        sub += 1
    return basename


def MainProgram(args):
    # Check mode
    mode = args.mode
    if mode == "record" or mode == "r":
        print("Recording...")

        if args.file != None:  # Save file passed
            # Create save dir name from args.file
            saveDir = args.file + ".sav"

            if not os.path.exists(saveDir):  # File/dir does not yet exist
                os.mkdir(saveDir)  # Create directory for saving
                # Create image directory
                os.mkdir(os.path.join(saveDir, "image/"))

                try:
                    Record(saveDir, os.path.join(saveDir, "image/"))
                except KeyboardInterrupt:
                    print("Saving...")
                    # Add .zip to name
                    zfName = args.file + ".zip"

                    # Compress save dir. Relpath is used to begin zipping inside the save dir structure.
                    zf = zipfile.ZipFile(zfName, "w")
                    topDir = os.path.join(saveDir, "..")
                    for dirname, subdirs, files in os.walk(saveDir):
                        if dirname != saveDir:
                            zf.write(dirname, os.path.relpath(
                                dirname, saveDir))
                        for file in files:
                            zf.write(os.path.join(dirname, file), os.path.join(
                                os.path.relpath(dirname, saveDir), os.path.basename(file)))
                    zf.close()

                # Delete save dir
                shutil.rmtree(saveDir, ignore_errors=True)
            else:
                print("[!] Passed file/directory already exists!")
        else:
            print("[!] No save file passed!")

    elif mode == "playback" or mode == "p":
        print("Playback")
        if args.file != None:  # File to load is passed
            if os.path.isfile(args.file):
                # Assign workDir as needed. If not specified, name based on loaded zip file.
                workDir = args.workdir
                if workDir == None:
                    basename = GetValidNewDirFromFilename(args.file)
                    workDir = os.path.join(
                        os.path.dirname(args.file), basename)
                else:
                    if not IsDirEmpty(workDir):  # Create a subdirectory
                       workDir = os.path.join(workDir, GetValidNewDirFromFilename(args.file))

                # Extract zip file
                with zipfile.ZipFile(args.file, "r") as zf:
                    zf.extractall(workDir)

            else:
                print("[!] Given playback file does not exist!")
        else:
            print("[!] No file passed to load for playback!")

    else:
        print("[!] Unknown mode: {}! Available modes: record, playback.".format(mode))

# Main content begins
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="RealSense Skeleton Tracking.")
    parser.add_argument(
        "mode", type=str, help="Program mode (record, playback)")
    parser.add_argument("-f", "--file", type=str,
                        help="File to save or load (depends on current mode).")
    parser.add_argument("-w", "--workdir", type=str,
                        help="Working directory to extract playback files to. By default creates a directory where the playback is stored.")

    args=parser.parse_args()

    MainProgram(args)  # Run main program
    print("Quiting...")
