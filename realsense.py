#!/usr/bin/env python3
import util as cm
import cv2
import argparse
import pickle
import os
import shutil
import json
import tarfile
import re
import time, datetime
import render, record
import settings
from multiprocessing import Process
import queue

def IsDirEmpty(dir):
    if not os.path.isdir(dir):
        return True
    
    return len(os.listdir(dir)) == 0


def GetValidNewDirFromFilename(fullPath):
    basename = os.path.basename(fullPath)
    if basename[-len(settings.data_file_extension):] == settings.data_file_extension:  # Check if it has the data file extension
        basename = basename[:-4]
    else:
        basename = basename + "_extract"
        
    # String digits to the end. Not very clean though.
    sub = 0
    while os.path.isdir(os.path.join(os.path.dirname(fullPath), basename)):
        basename += "{}".format(sub)
        sub += 1
    return basename

class PlaybackFrame:
    def __init__(self, color_image, skeletons, depth, depth_intrinsic):
        self.color_image = color_image
        self.skeletons = skeletons
        self.depth = depth
        self.depth_intrinsic = depth_intrinsic

def GetPickledAtFrame(pickleFile, frame_number):
    # Starting at next pickle, loop until its found
    found = False
    p = None
    while not found:
        try:
            p = pickle.load(pickleFile)
        except EOFError:
            return None
        
        if p[0] == frame_number:
            found = True
    
    if not found:
        return None
    else:
        return p[1]

def LoadPlaybackData(in_queue, out_frames):
    while True:
        frame_number, image_loc, skeletons_file, depth_file, intrinsic_file = in_queue.get()
        
        if frame_number == "Quit": # Quit signal
            in_queue.task_done()
            break
        
        # Get color image
        color_image = cv2.imread(image_loc)
        
        # Get skeletons
        skeletons = GetPickledAtFrame(skeletons_file, frame_number)
        
        # Get depth
        depth = GetPickledAtFrame(depth_file, frame_number)
        
        # Get depth_intrinsic
        depth_intrinsic = GetPickledAtFrame(intrinsic_file, frame_number)
        
        # Return frame
        out_frames[frame_number] = PlaybackFrame(color_image, skeletons, depth, depth_intrinsic)
        
        in_queue.task_done()

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

                # Create save files
                pickleFileLoc = os.path.join(saveDir, "save.skel")
                pickleFile = open(pickleFileLoc, "wb")
                depthFileLoc = os.path.join(saveDir, "depth.sav")
                depthPickle = open(depthFileLoc, "wb")
                depthIntrFileLoc = os.path.join(saveDir, "depth_intrinsic.sav")
                depthIntrPickle = open(depthIntrFileLoc, "wb")
                settingsFileLoc = os.path.join(saveDir, "settings.json")
                settingsFile = open(settingsFileLoc, "w")
                frameDataLoc = os.path.join(saveDir, "frame_data.json")
                frame_data = {} # Frame data is returned by record and must be written here
                
                try: 
                    record.Record(saveDir, os.path.join(saveDir, "image/"), pickleFile, depthPickle, depthIntrPickle, settingsFile, frame_data)
                except KeyboardInterrupt:
                    print("Saving...")
                    
                    # Write the frame data to file
                    with open(frameDataLoc, 'w') as fd:
                        json.dump(frame_data, fd)
                    
                    # Close the files
                    pickleFile.close()
                    depthPickle.close()
                    depthIntrPickle.close()
                    settingsFile.close()
                    
                    # Compress the directory                    
                    tar = tarfile.open(args.file + settings.data_file_extension, "w:gz", compresslevel=settings.data_compress_level)
                    tar.add(saveDir, arcname="")
                    tar.close()

                # Delete save dir
                shutil.rmtree(saveDir, ignore_errors=True)
            else:
                print("[!] Passed file/directory already exists!")
        else:
            print("[!] No save file passed!")

    elif mode == "playback" or mode == "p":
        print("Playback")
        if args.file != None:  # File to load is passed
            playbackFile = args.file
            if not os.path.isfile(playbackFile):
                playbackFile = args.file + settings.data_file_extension
                
            if os.path.isfile(playbackFile):
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
                with tarfile.open(playbackFile, "r:gz") as tar:
                    tar.extractall(workDir)
                    
                # Read joint_confidence
                joint_confidence = 0.2
                with open(os.path.join(workDir, "settings.json"), "r") as jf:
                    data = json.load(jf)
                    if "joint_confidence" in data:
                        joint_confidence = data["joint_confidence"]
                    
                # Read frame_data
                frame_data = {}
                with open(os.path.join(workDir, "frame_data.json"), "r") as fd:
                    frame_data = json.load(fd)
                
                # Sorting image based on number
                def atoi(text):
                    return int(text) if text.isdigit() else text
            
                def human_key(text):
                    return [atoi(c) for c in re.split(r'(\d+)', text)]
                
                # Open files
                imageDir = os.path.join(workDir, "image/")
                sortedImages = os.listdir(imageDir)
                sortedImages.sort(key=human_key)
                skels = open(os.path.join(workDir, "save.skel"), "rb")
                depth_file = open(os.path.join(workDir, "depth.sav"), 'rb')
                intrinsic_file = open(os.path.join(workDir, "depth_intrinsic.sav"), 'rb')
                
                # Load frames with multiprocessing
                loaded_frames = {}
                load_queue = queue.Queue()
                last_queued_frame_number = 0
                total_frames = len(sortedImages)
                LOAD_AHEAD = 3
                playback_done = False
                
                # Preload
                while last_queued_frame_number < LOAD_AHEAD:
                    load_queue.put((last_queued_frame_number, 
                                    os.path.join(imageDir, sortedImages[last_queued_frame_number]),
                                    skels, depth_file, intrinsic_file))
                    last_queued_frame_number += 1
                
                import threading
                p = threading.Thread(target=LoadPlaybackData, args=(load_queue, loaded_frames))
                p.start()
                
                # Wait til loading min preload frames
                while len(loaded_frames) < LOAD_AHEAD:
                    print("Wai                playback_done = Trueting {}".format(len(loaded_frames)))
                    time.sleep(1)
                
                '''color_images = {}
                skeletons_frames = {}
                depth_frames = {}
                depth_intrinsic_frames = {}'''
                '''for imageLoc in sortedImages:
                    imageCounter = human_key(imageLoc)[1]
                    color_images[imageCounter] = cv2.imread(os.path.join(imageDir, imageLoc))
                    
                    skeletons = None
                    with open(os.path.join(workDir, "save.skel"), 'rb') as skels:
                        found = False
                        while not found:
                            p = None
                            try:
                                p = pickle.load(skels)
                            except EOFError:
                                break
                            
                            if p[0] == imageCounter:
                                skeletons = p[1]
                                found = True
                    
                    if skeletons == None:
                        print("No skeleton found for frame: {}".format(imageCounter))
                    else:
                        skeletons_frames[imageCounter] = skeletons
                                
                    depth = None
                    with open(os.path.join(workDir, "depth.sav"), 'rb') as depthFile:
                        found = False
                        while not found:
                            p = None
                            try:
                                p = pickle.load(depthFile)
                            except EOFError:
                                break
                            
                            if p[0] == imageCounter:
                                depth = p[1]
                                found = True
                                
                    if depth == None:
                        print("No depth found for frame: {}".format(imageCounter))
                    else:
                        depth_frames[imageCounter] = depth
                                
                    depth_intrinsic = None
                    with open(os.path.join(workDir, "depth_intrinsic.sav"), 'rb') as df:
                        found = False
                        while not found:
                            p = None
                            try:
                                p = pickle.load(df)
                            except EOFError:
                                break
                            
                            if p[0] == imageCounter:
                                depth_intrinsic = p[1]
                                found = True
                                
                    if depth_intrinsic == None:
                        print("No depth intrinsic found for frame: {}".format(imageCounter))
                    else:
                        depth_intrinsic_frames[imageCounter] = depth_intrinsic'''
                        
                    
                # Render each frame
                print("Playing...")
                cv2.namedWindow(settings.window_name_playback, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO) # Create Window
                last_time = datetime.datetime.now() # Last render time
                #for frame_number in color_images:
                print("Total frames: ", total_frames)
                for frame_number in range(total_frames):
                    while frame_number not in loaded_frames:
                        print("Frame {} not loaded yet...".format(frame_number))
                    
                    # Frame loaded, play now
                    frame = loaded_frames[frame_number]
                    color_image = frame.color_image
                    skeletons = frame.skeletons
                    depth = frame.depth
                    depth_intrinsic = frame.depth_intrinsic
                    
                    # Verify all data is here
                    play = True
                    if color_image is None:
                        print("No image for frame {}".format(frame_number))
                        play = False
                    if skeletons is None:
                        print("No skeletons for frame {}".format(frame_number))
                        play = False
                    if depth is None:
                        print("No depth for frame {}".format(frame_number))
                        play = False
                    if depth_intrinsic is None:
                        print("No depth_intrinsic for frame {}".format(frame_number))
                        play = False
                    
                    if play:
                        # render the skeletons on top of the acquired image and display it
                        cm.render_result(skeletons, color_image, joint_confidence)
            
                        render.render_ids_3d(
                            color_image, skeletons, depth, depth_intrinsic, joint_confidence
                        )
                        cv2.imshow(settings.window_name_playback, color_image)
                        
                        if cv2.waitKey(1) == 27:
                            break
                    else:
                        print("Could not play.")
                    
                    del loaded_frames[frame_number]
                    while load_queue.qsize() < min(LOAD_AHEAD, total_frames - last_queued_frame_number ):
                        load_queue.put((last_queued_frame_number, 
                                    os.path.join(imageDir, sortedImages[last_queued_frame_number]),
                                    skels, depth_file, intrinsic_file))
                        last_queued_frame_number += 1
                    
                    # Get delta time and wait if necessary
                    frame_str = "{}".format(frame_number)
                    if frame_str in frame_data:
                        now = datetime.datetime.now()
                        delta = now - last_time
                        delta = delta.total_seconds()
                        last_time = now
                        
                        record_delta = frame_data[frame_str]["Delta Time"]
                        if record_delta > delta:
                            time.sleep(record_delta - delta)
                    else:
                        print("No frame data found for frame: {}".format(frame_number))
                    
                    '''if frame_number in color_images and frame_number in skeletons_frames and frame_number in depth_frames and frame_number in depth_intrinsic_frames:
                        color_image = color_images[frame_number]
                        skeletons = skeletons_frames[frame_number]
                        depth = depth_frames[frame_number]
                        depth_intrinsic = depth_intrinsic_frames[frame_number]
                    
                        # render the skeletons on top of the acquired image and display it
                        cm.render_result(skeletons, color_image, joint_confidence)
            
                        render.render_ids_3d(
                            color_image, skeletons, depth, depth_intrinsic, joint_confidence
                        )
                        cv2.imshow(settings.window_name_playback, color_image)
                        
                        if cv2.waitKey(1) == 27:
                            break
                    
                    # Get delta time and wait if necessary
                    frame_str = "{}".format(frame_number)
                    if frame_str in frame_data:
                        now = datetime.datetime.now()
                        delta = now - last_time
                        delta = delta.total_seconds()
                        last_time = now
                        
                        record_delta = frame_data[frame_str]["Delta Time"]
                        if record_delta > delta:
                            time.sleep(record_delta - delta)
                    else:
                        print("No frame data found for frame: {}".format(frame_number))'''

                load_queue.put(("Quit", None, None, None, None))
                load_queue.join()
                print("Waiting on thread...")
                p.join()
                cv2.destroyAllWindows()
                
                # Close data files
                skels.close()
                depth_file.close()
                intrinsic_file.close()
                
                # Delete save dir
                shutil.rmtree(workDir, ignore_errors=True)
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
