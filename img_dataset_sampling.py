import os
from random import randint
import shutil
# root_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/A/",
# 			"/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/B/"]
# target_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/A/",
# 				"/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/B/"]
# root_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/B/"]
# target_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/B/"]
source_dir = ["/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/no_glasses_merged/",
                "/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/with_glasses_merged/"]
target_dir = ["/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/no_glasses_sampled/",
                "/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4/with_glasses_sampled/"]

sampling_rate = 5 # choosing 1 image in 10
# moving_number = 100
for i in range(len(source_dir)):
    one_source = source_dir[i]
    one_target = target_dir[i]

    try:
        os.mkdir(one_target)
    except:
        print("Folder exists")

    files = os.listdir(one_source)
    for j in range(len(files)):
        if j % sampling_rate == 0:
            file_to_move = files[j]
            shutil.copy(one_source+file_to_move, one_target+file_to_move)
