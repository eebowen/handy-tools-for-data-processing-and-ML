import os
from random import randint
import shutil
# root_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/A/",
# 			"/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/B/"]
# target_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/A/",
# 				"/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/B/"]
# root_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/B/"]
# target_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/B/"]
source_dir = ["/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/no_glasses",
                "/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/with_glasses"]
target_dir = ["/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/no_glasses_merged",
                "/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/with_glasses_merged"]


for i in range(len(source_dir)):
    one_source = source_dir[i]
    one_target = target_dir[i]

    try:
        os.mkdir(one_target)
    except:
        print("Folder exists")

    # r=root, d=directories, f = files
    print("start copying")
    for r, d, f in os.walk(one_source):
        for file in f:
            if '.jpg' in file:
                file_path = os.path.join(r, file)
                shutil.copy(file_path, one_target+"/"+file)
