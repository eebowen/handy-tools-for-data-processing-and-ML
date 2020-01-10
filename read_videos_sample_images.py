# read video, rotate 180, and sample images
import os
from glob import glob 
import shutil
import cv2
import numpy as np

foot_video_root = '/mnt/cvrr-nas/WorkArea3/WorkArea3_NoBackup/Userarea/bowen/lisat_foot_data/raw_videos/'
foot_image_root = '/mnt/cvrr-nas/WorkArea3/WorkArea3_NoBackup/Userarea/bowen/lisat_foot_data/raw_images/'

classes = os.listdir(foot_video_root)

# sample_rate = 15 # pick one image in every 15 images
# for one_class in classes:
#     one_class_video_root = os.path.join(foot_video_root,one_class)
#     print(one_class_video_root)
#     one_class_image_root = os.path.join(foot_image_root,one_class)
#     print(one_class_image_root)
#     for one_video_path in glob(one_class_video_root+'/*'):
#         cap = cv2.VideoCapture(one_video_path)
#         ret, frame = cap.read()
#         print(frame.shape)
#         h,w,_ = frame.shape
#         frame_count = 0
#         rotated180 = (w / 2, h / 2)
#         while(cap.isOpened()):
#             ret, frame = cap.read()
#             if frame_count % sample_rate == 0:
#                 M = cv2.getRotationMatrix2D(center, 180, 1.0)
#                 rotated180 = cv2.warpAffine(frame, M, (w, h))

#             cv2.imshow('frame',frame)
#             frame_count += 1
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()




#test
import cv2 
import numpy as np 
   
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('/mnt/cvrr-nas/WorkArea3/WorkArea3_NoBackup/Userarea/bowen/lisat_foot_data/raw_videos/On Brake/3D_L8598.MP4') 
   
# Check if camera opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 
   
# Read until video is completed 
while(cap.isOpened()): 
      
  # Capture frame-by-frame 
  ret, frame = cap.read() 
  if not ret:
    print('wrong')
  if ret == True: 
   
    # Display the resulting frame 
    cv2.imshow('Frame', frame) 
   
    # Press Q on keyboard to  exit 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break
   
  # Break the loop 
  else:  
    break
   
# When everything done, release  
# the video capture object 
cap.release() 
   
# Closes all the frames 
cv2.destroyAllWindows() 