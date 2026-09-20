import os
import logging
from argparse import ArgumentParser
import shutil

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--mask_path", "-m", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
# Sequential matching walks images in filename order. A stereo clip interleaves the two
# cameras under one numbering, so a given overlap spans half as much time as it would
# for a monocular clip; raise it to keep the same temporal reach.
parser.add_argument("--overlap", type=int, default=0,
                    help="SequentialMatching.overlap; 0 leaves COLMAP's default.")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
glomap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "glomap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# COLMAP 3.12 moved the backend-agnostic knobs (use_gpu, max_num_matches) out of the
# SIFT-specific option groups into FeatureExtraction/FeatureMatching, and rejects the
# old spelling outright. Probe rather than pin, so this runs on either side of 3.12.
def _opt_group(subcommand, new, old):
    import subprocess
    try:
        proc = subprocess.run([colmap_command.strip('"'), subcommand, "-h"],
                              capture_output=True, text=True)
    except OSError:
        return old
    # COLMAP prints the option list on stderr, help banner on stdout; read both.
    return new if f"--{new}.use_gpu" in (proc.stdout + proc.stderr) else old

EXTRACT_GROUP = _opt_group("feature_extractor", "FeatureExtraction", "SiftExtraction")
MATCH_GROUP = _opt_group("sequential_matcher", "FeatureMatching", "SiftMatching")

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
    ## Feature extraction
    # Read from inputs/, not images/: images/ is where image_undistorter writes its
    # output at the end of this script, and COLMAP's copy step throws if the
    # destination file already exists. inputs/ holds the raw frames throughout.
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/inputs \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --" + EXTRACT_GROUP + ".use_gpu " + str(use_gpu)
    
    if args.mask_path is None:
        args.mask_path = args.source_path + "/masks"
        
    if os.path.exists(args.mask_path):
        print(f"Using mask path: {args.mask_path}")
        feat_extracton_cmd += " --ImageReader.mask_path " + args.mask_path
        
    print(f"Executing: {feat_extracton_cmd}")
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " sequential_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --" + MATCH_GROUP + ".use_gpu " + str(use_gpu) + \
        f" --{MATCH_GROUP}.max_num_matches 16384"
    if args.overlap > 0:
        feat_matching_cmd += f" --SequentialMatching.overlap {args.overlap}"
    print(f"Executing: {feat_matching_cmd}")
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (glomap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/inputs \
        --output_path "  + args.source_path + "/distorted/sparse"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/inputs \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")
    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print(f"Finished processing {args.source_path}")
