#!/usr/bin/env python3
import argparse
import os
import shutil
import json
import tarfile
import record
import playback
import settings


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
                frame_data = {}  # Frame data is returned by record and must be written here

                try:
                    record.Record(saveDir, os.path.join(
                        saveDir, "image/"), pickleFile, depthPickle, depthIntrPickle, settingsFile, frame_data)
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
                    tar = tarfile.open(args.file + settings.data_file_extension,
                                       "w:gz", compresslevel=settings.data_compress_level)
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
                playback.Playback(playbackFile, args.workdir, args.file)
            else:
                print("[!] Given playback file does not exist!")
        else:
            print("[!] No file passed to load for playback!")

    else:
        print("[!] Unknown mode: {}! Available modes: record, playback.".format(mode))


# Handle arguments and call main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RealSense Skeleton Tracking.")
    parser.add_argument(
        "mode", type=str, help="Program mode (record, playback)")
    parser.add_argument("-f", "--file", type=str,
                        help="File to save or load (depends on current mode).")
    parser.add_argument("-w", "--workdir", type=str,
                        help="Working directory to extract playback files to. By default creates a directory where the playback is stored.")

    args = parser.parse_args()

    MainProgram(args)  # Run main program
    print("Quiting...")
