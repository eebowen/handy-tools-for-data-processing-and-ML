from glob import glob
import numpy as np
import os
import cv2
import pandas as pd
import time
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/ridel/bowen/caffe/python')
import caffe


video_path = '/mnt/hdd/lisa_t_data/gaze/*'

subjects = glob(video_path)
# print(subjects)
# print(subjects)
list_gaze_class = ['Eyes Closed', 'Forward', 'Left Shoulder', 'Left Mirror', 'Lap', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror', 'Right Shoulder']

list_gaze_class_dir_name = ['eyes_closed', 'forward', 'left_shoulder', 'left_mirror', 'lap', 'speedometer', 'radio', 'rearview', 'right_mirror', 'right_shoulder']
list_gaze_class_dir_name = ['eyes_closed', 'forward', 'shoulder', 'left_mirror', 'lap', 'speedometer', 'radio', 'rearview', 'right_mirror']
test_subject = 'bowen'
test_subject = 'hanisha'
# some_subjects = ['/mnt/hdd/lisa_t_data/gaze/hanisha', '/mnt/hdd/lisa_t_data/gaze/steve']
some_subjects = ['/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v3/akshay', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v3/nachiket', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v3/professor', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v3/hanisha', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v3/bowen', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v3/steve']

some_subjects = ['/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v4/akshay', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v4/nachiket', \
'/mnt/hdd/bowen/data_folder/lisa_t_data/gaze_data_v4/larry']


def convert_to_mp4():
	

	write_file = open("convert_to_mp4.sh","w+")

	# this is to save the label file
	# for subject in subjects:
	for subject in some_subjects:
		# print(test_subject in subject)
		# if test_subject in subject:

		subject = subject + '/*'
		gaze_zones = glob(subject)
		print(gaze_zones)
		# for one subject
		for gaze_zone in gaze_zones:
			# print(gaze_zone)
			one_video_path = os.path.join(gaze_zone, 'IR_1.avi')
			one_video_output = os.path.join(gaze_zone, 'IR_1.mp4')
			line = 'ffmpeg -i ' + one_video_path + ' -acodec copy -vcodec copy ' + one_video_output +'\n'
			write_file.write(line)
			# find the gaze class for every dir
			class_label = np.argmax([one_class in gaze_zone for one_class in list_gaze_class_dir_name])


			# print(list_gaze_class_dir_name[class_label])

			# read the frame label file and find out which frame to process

			# run pose net on the file 

			# crop the image 
			# save the bounding box
			# write file one_vidio_path to train and val photo

		# else:
		# 	print('test subject!!!!!!!!!!!!!!!!!!!!!!!!!')
	write_file.close()

def write_label_file():
	# delete this line if write every subject to the file.
	
	write_file = open("lisat_label_data_v4.txt","w+")
	i = 0
	# for subject in subjects:
	for subject in some_subjects:

		# print(test_subject in subject)
		# if test_subject in subject:

		subject = subject + '/*'
		gaze_zones = glob(subject)

		# for one subject
		for gaze_zone in gaze_zones:
			# print(gaze_zone)
			one_video_path = os.path.join(gaze_zone, 'IR_1.mp4')
			# one_video_output = os.path.join(gaze_zone, 'IR_1.mp4')
			# line = 'ffmpeg -i ' + one_video_path + ' -acodec copy -vcodec copy ' + one_video_output +'\n'
			cap = cv2.VideoCapture(one_video_path)
			frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

			class_label = np.argmax([one_class in gaze_zone for one_class in list_gaze_class_dir_name])
			line = one_video_path + ' ' + str(int(class_label)) + ' 0 ' + str(int(frame_count_total)) + '\n'
			write_file.write(line)
			# find the gaze class for every dir
	write_file.close()


# open videos and check the total frame numbers
def check_video_frames():
	all_video_path = pd.read_csv('./lisat_frame_label_h_s.txt', delim_whitespace=True, names=('path', 'label', 'start', 'end'))
	for i in range(all_video_path.shape[0]):
		one_video_path = all_video_path['path'][i]
		print(one_video_path)
	# # one_video_path = '/mnt/cvrr-nas/realsense/gaze/bowen/gaze_bowen_w_radio/IR_1.mp4'
		cap = cv2.VideoCapture(one_video_path)
		frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		print(frame_count_total)
		# for 
		start_frame = 0
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		frame_cnt = start_frame
		while(cap.isOpened()):

			print(frame_cnt)
			im_captured, im_raw = cap.read()
			if im_captured is False:
				break
			cv2.namedWindow(one_video_path, cv2.WINDOW_NORMAL)
			cv2.imshow(one_video_path, im_raw)
		    # cv2.resizeWindow(one_video_path, 640, 480)
			if cv2.waitKey(1) and 0xFF == ord('q'):
				break
			frame_cnt += 1
			# time.sleep(0.1)

		cap.release()
		cv2.destroyAllWindows()



def frame_cnt_for_all_videos():
	all_video_path = pd.read_csv('./lisat_frame_label.txt', delim_whitespace=True, names=('path', 'label', 'start', 'end'))





# this is to read the label file videos and output the bbox and train.txt and val.txt
def read_label_files():
	with open('./lisat_frame_label_new.txt') as f:
		lines = [line for line in f]
		print(lines[0])
		# for line in f:
		# 	data = line.split()
		# 	one_video_path = data[0]
		# 	print(one_video_path)
		# 	gaze_zone = data[1]
		# 	start_frame = int(data[2])
		# 	end_frame = int(data[3])

		# 	frame_cnt = start_frame
        
# results visualization 
def draw_log_files():
	log = pd.read_csv('./lisat_training_files_v1/results_v1/v1_train.log')

import os
def hist_equalization():
	train_data = True
	if train_data:
		images = glob("./lisat_training_files_v4/train_v4/*")
		for image in images:
			# print(image)
			img = cv2.imread(image)
			equ = cv2.equalizeHist(img[:, :, 0])
			# equ = cv2.equalizeHist(img)
			out_img_folder = './lisat_training_files_v4/train_v4_equ/'
			loc = 35 # 35
			out_img_path = os.path.join(out_img_folder, image[loc:])
			print(out_img_path)
			cv2.imwrite(out_img_path,equ)
	else:		
		images = glob("./lisat_training_files_v4/val_v4/*")
		for image in images:
			# print(image)
			img = cv2.imread(image,0)
			equ = cv2.equalizeHist(img)
			out_img_folder = './lisat_training_files_v4/val_v4_equ/'
			out_img_path = os.path.join(out_img_folder, image[33:])
			print(out_img_path)
			try:
				os.mkdir(out_img_folder)
			except:
				print('val folder exits')
			cv2.imwrite(out_img_path,equ)
			# hist,bins = np.histogram(img.flatten(),256,[0,256])

	# cdf = hist.cumsum()
	# cdf_normalized = cdf * hist.max()/ cdf.max()

	# plt.plot(cdf_normalized, color = 'b')
	# plt.hist(img.flatten(),256,[0,256], color = 'r')
	# plt.xlim([0,256])
	# plt.legend(('cdf','histogram'), loc = 'upper left')
	

	# equ = cv2.equalizeHist(img)
	# res = np.hstack((img,equ)) #stacking images side-by-side
	# cv2.imshow('two images', res)
	# cv2.waitKey()
	# cv2.imwrite('res.jpg',res)
	# plt.show()



def read_mean_file_and_save():

	data_root = '/mnt/hdd/bowen/lisa/cabin_2.0/lisat_gaze_data/lisat_training_files_v4'
	mean_file_path = os.path.join(data_root, 'gaze_mean_train_v4_all.binaryproto')
	mean_file = open(mean_file_path, 'rb').read()
	blob = caffe.proto.caffe_pb2.BlobProto()
	blob.ParseFromString(mean_file)
	mean_gaze = caffe.io.blobproto_to_array(blob)
	mean_npy_path = os.path.join(data_root, 'gaze_mean_train_v4_all.npy')
	np.save(mean_npy_path, mean_gaze)


# convert_to_mp4()
# write_label_file()
# check_video_frames()

# read_label_files()
# hist_equalization()
read_mean_file_and_save()
#