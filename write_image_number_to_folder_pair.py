import os
from random import randint
import shutil


data_root = "/mnt/hdd/bowen/data_folder/lisa_t_data/image_data/lisat_img_data_v5/"
folders = ['with_glasses_night', 'no_glasses_night', 'no_glasses_day', 'with_glasses_day']

for r, d, f in os.walk(data_root):
        for file in f:
            if '.jpg' in file:
                print(r)




# for one_folder in folders:
# 	os.path.join(data_root,one_folder)


# one_dir = source_dir[i]
# one_target = target_dir[i]

# try:
# 	os.mkdir(one_target)
# except:
# 	print("Folder exists")

# files = os.listdir(one_dir)
# moving_num = int(len(files) / moving_percent)

# all_random = [randint(0, len(files)) for i in range(moving_num)]
# no_repear_random = set(all_random)
# print(f"Cutting dir: {one_dir} \nLen: {len(files)} \nMoving nummber: {moving_num}")
# for j in no_repear_random:
# 	file_to_move = files[j]
# 	# print(one_dir+file_to_move)
# 	shutil.move(one_dir+file_to_move, one_target+file_to_move)
