import os
from random import randint
import shutil
source_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/A/",
			"/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/B/"]
target_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/A/",
				"/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/B/"]
# source_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/train/B/"]
# target_dir = ["/mnt/hdd/bowen/glasses_removal/PyTorch-CycleGAN/datasets/lisat_gaze/test/B/"]
source_dir = ["/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4_sampled/train/A/",
            "/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4_sampled/train/B/"]
target_dir = ["/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4_sampled/test/A/",
                "/home/cvrr/workspace/cycleGAN/dataset/lisat_gaze_v4_sampled/test/B/"]

moving_percent = 10
# moving_number = 100
for i in range(len(source_dir)):
	one_dir = source_dir[i]
	one_target = target_dir[i]
	
	try:
		os.mkdir(one_target)
	except:
		print("Folder exists")
	
	files = os.listdir(one_dir)
	moving_num = int(len(files) / moving_percent)
	
	all_random = [randint(0, len(files)) for i in range(moving_num)]
	no_repear_random = set(all_random)
	print(f"Cutting dir: {one_dir} \nLen: {len(files)} \nMoving nummber: {moving_num}")
	for j in no_repear_random:
		file_to_move = files[j]
		# print(one_dir+file_to_move)
		shutil.move(one_dir+file_to_move, one_target+file_to_move)
