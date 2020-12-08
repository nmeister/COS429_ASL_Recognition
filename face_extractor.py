# Import libraries
import os
import cv2
import numpy as np

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Create directory 'faces' if it does not exist
if not os.path.exists('val-faces'):
	print("New directory created")
	os.makedirs('val-faces')
SIGNS = ['ALWAYS', 'BEAUTIFUL', 'BIG', 'BORED', 'BORN', 'BREAK', 'BREAKDOWN', 'CANCEL', 'CANNOT', 'CHAT','CLEAN','CORRECT', 'CRASH', 'FAVORITE', 'FRIEND', 'HAPPY','LIVE','LOVE','NO','NONE','PARTY','SICK','THANK_YOU','WHAT','WHY']
#SIGNS = ['NO']
# Loop through all images and strip out faces
count = 0
sign_count = 0 
all_signs = 'val'
for sign in os.listdir(base_dir + all_signs):
	print(sign)
	if sign not in SIGNS:
		print('continued for', sign)
		continue 
	sign_count += 1
	for video_folder in os.listdir(base_dir + all_signs + '/' + sign):
		if not video_folder.isdigit() or not os.path.isdir(base_dir + all_signs + '/' + sign + '/' + video_folder):
			continue
		for file in os.listdir(base_dir + all_signs + '/' + sign + '/' + video_folder):
			file_name, file_extension = os.path.splitext(file)
			if (file_extension in ['.png','.jpg']):
				image = cv2.imread(base_dir + all_signs + '/' + sign + '/' + video_folder + '/' + file)

				(h, w) = image.shape[:2]
				blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

				model.setInput(blob)
				detections = model.forward()

				# Identify each face
				for i in range(0, detections.shape[2]):
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					confidence = detections[0, 0, i, 2]

					# If confidence > 0.5, save it as a separate file
					if (confidence > 0.5):
						#print(sign + 'detected')
						count += 1
						frame = image[startY:endY, startX:endX]
						if not os.path.exists('val-faces/' + sign + '/' + video_folder):
							# print("New directory created")
							os.makedirs('val-faces/' + sign + '/' + video_folder)
						cv2.imwrite(base_dir + 'val-faces/' + sign + '/' + video_folder + '/' + 'face-' + file, frame)

print("Extracted " + str(count) + " faces from all images")
print('signs total', sign_count)