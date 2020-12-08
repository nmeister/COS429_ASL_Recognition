# Import libraries
import os
import cv2
import numpy as np
from crop import crop_center as cc

# Define paths
base_dir = os.path.dirname(__file__)
print(base_dir)
dirname = 'combine'
if not os.path.exists(dirname):
	print("New directory created")
	os.makedirs(dirname)

SIGNS = ['ALWAYS', 'BEAUTIFUL', 'BIG', 'BORED', 'BORN', 'BREAK', 'BREAKDOWN', 'CANCEL', 'CANNOT', 'CHAT','CLEAN','CORRECT', 'CRASH', 'FAVORITE', 'FRIEND', 'HAPPY','LIVE','LOVE','NO','NONE','PARTY','SICK','THANK_YOU','WHAT','WHY']

count = 0
signs_count = 0 
seen = 0 
folder = 'images'
folder_type = ['/train', '/val']
#folder_type = ['/val']
input_size = 224
for ft in folder_type:
	print('folder type:', ft)
	curr_path = base_dir + folder + ft
	for sign in os.listdir(curr_path):
		print(sign)
		if sign not in SIGNS:
			print('continued for', sign)
			continue
		video_count = 0 
		for video_folder in os.listdir(curr_path + '/' + sign):
			if not video_folder.isdigit() or not os.path.isdir(curr_path + '/' + sign + '/' + video_folder):
				continue
			prev = None
			file_count = 0 
			for file in os.listdir(curr_path + '/' + sign + '/' + video_folder):
				file_name, file_extension = os.path.splitext(file)
				if file == '.DS_Store' or file_name == '.DS_Store':
					print('.DS_STORE')
					continue
				# images/train/CANCEL/314999/314999-10.jpg
				image_path = curr_path + '/' + sign + '/' + video_folder + '/' + file
				print(image_path)
				if (file_extension in ['.png','.jpg']):
					image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
				(h, w) = image.shape[:2]
				image = cv2.resize(image, (input_size, input_size))
				image = cc(image)
				face_path_arr = image_path.split('/')
				face_path_arr[0] = 'faces'
				face_path_arr[-1] = 'face-'+ face_path_arr[-1]
				face_path = "/".join(face_path_arr)
				face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
				if face is None and prev:
					print('face is none')
					face_path = prev
					face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
				else:
					prev = face_path
				face = cv2.resize(face, (input_size // 2, input_size))
				image = cv2.resize(image, (input_size // 2, input_size))
				image = cv2.hconcat([image, face])
				final_path = image_path.split('/')
				final_path[0] = 'combine'
				final_path = "/".join(final_path)
				if not os.path.exists('combine/' + ft + '/' + sign + '/' + video_folder):
					os.makedirs('combine/' + ft + '/' + sign + '/' + video_folder)
				cv2.imwrite(final_path, image)
				file_count += 1
			print('file counted for', sign, '-', video_folder, file_count)
			video_count += 1
		print('video count:', video_count)
		signs_count += 1
	print('total signs in', ft, ':', signs_count)