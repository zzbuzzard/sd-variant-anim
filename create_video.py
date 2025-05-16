import argparse
import os
from os.path import join
from PIL import Image

import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--directory", type=str, required=True, help="Folder containing images.")
    parser.add_argument("-f", "--fps", type=int, default=10, help="Frames per second.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output path for MP4. Defaults to 'video.mp4' inside input directory.")
    args = parser.parse_args()

    if args.out is None:
        args.out = join(args.directory, "video.mp4")

    assert os.path.isdir(args.directory), f"Could not find directory '{args.directory}'"
    assert os.path.isfile(util.output_path(args.directory, 0)), f"Could not find image 0000000.jpg in directory '{args.directory}'"

    writer = util.AvVideoWriter(512, 512, args.out, args.fps)

    print("Beginning...")
    ctr = 0
    while os.path.isfile(util.output_path(args.directory, ctr)):
        image = Image.open(util.output_path(args.directory, ctr))
        writer.write(image)
        ctr += 1

    writer.close()
    print("Completed. Saved to", args.out)
