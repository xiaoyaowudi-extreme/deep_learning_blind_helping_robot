#===========================================#
#                                           #
#                                           #
#--------IMAGE RECOGNITION SERVER-----------#
#----------WRITTEN BY Ziyao Xiao------------#
#-----------------2018 (c)------------------#
#                                           #
#                                           #
#===========================================#

#Copyright by Ziyao Xiao, 2018 (c)

#Licensed under the MIT License:

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import os
from flask import Flask,request
import uuid
import recognition
import cv2
import config
import numpy as np
import zebra_crossing_detection
import subprocess

fast_calculate = cv2.imread("./fast_calculate.jpg")

#app
app = Flask(__name__)

#tensorflow image classification
classification = recognition.recognition(config.height, config.width, "./graph.pb", 0, use_gpu = False, fast_recognition = True, fast_recognition_image = fast_calculate)

def up_Gaus(x, y):
	if y%x==0:
		return int(y/x)
	else :
		return int(y/x +1)

def fit_to_scale(image):
	if image.shape[0] >= 500 and image.shape[1] >= 500:
		return
	after_height = image.shape[0] * max(up_Gaus(image.shape[0], 500), up_Gaus(image.shape[1], 500))
	after_width  = image.shape[1] * max(up_Gaus(image.shape[0], 500), up_Gaus(image.shape[1], 500))
	img = cv2.resize(image, (after_width, after_height), interpolation=cv2.INTER_CUBIC)
	return img

def expand_mask_dim(mask):
	mask_C3 = np.stack(( mask, mask, mask), axis = -1)
	return mask_C3

#recognition
@app.route('/recognition', methods = ['GET', 'POST'])
def recognize():
	try:
		if request.method != 'POST':
			return "<h1>ERROR</h1>"
		file = request.files['file']
		folder_name = os.path.join("uploads",str(uuid.uuid4()))
		os.system("mkdir "+folder_name)
		file_name = file.filename
		file_path = os.path.join(folder_name, file_name)
		file.save(file_path)
		img            = cv2.imread(file_path)
		detection_data = classification.recognize(img)
		needed_graph   = [ [ img * expand_mask_dim(detection_data['detection_masks'][i]), detection_data['detection_boxes'][i] ] for i in range(detection_data['num_detections']) if detection_data['detection_scores'][i] >= np.float32(0.50) and detection_data['detection_classes'][i] == 10 ]
		i=0
		good_status=[]
		for graph in needed_graph:
			actual_graph_small = graph[0][int(graph[1][0]*config.height):int(graph[1][2]*config.height), int(graph[1][1]*config.width):int(graph[1][3]*config.width)]
			actual_graph       = fit_to_scale(actual_graph_small)
			cv2.imwrite(os.path.join(folder_name, str(i)+".jpg"),actual_graph)
			output = subprocess.getoutput("./classify_server " + os.path.join(folder_name, str(i)+".jpg"))
			print(output)
			if "Back" in output:
				pass
			else:
				good_status.append([output,actual_graph_small])
			i=i+1
		print(good_status)
		if len(good_status) == 0:
			return "No sidewalk traffic light detected"
		else:
			good_status.sort( key = lambda x : x[1].shape[0] * x[1].shape[1])
			if "Red" in good_status[0][0]:
				return "Red sidewalk traffic light detected"
			else:
				return "Green sidewalk traffic light detected"
	except Exception:
		print("error occured")
		return "<h1>ERROR</h1>"
#zebra crossing
@app.route('/zebra-crossing', methods = ['GET', 'POST'])
def zerbra_crossing():
	try:
		if request.method != 'POST':
			return "<h1>ERROR</h1>"
		frame_1  = request.files['frame_1']
		frame_2  = request.files['frame_2']
		frame_3  = request.files['frame_3']
		frame_4  = request.files['frame_4']
		frame_5  = request.files['frame_5']
		frame_6  = request.files['frame_6']
		frames   = [ frame_1, frame_2, frame_3, frame_4, frame_5, frame_6 ]
		for i in range(6):
			folder_name = os.path.join("uploads_zebra",str(uuid.uuid4()))
			os.system("mkdir " + folder_name)
			frames[i].save(os.path.join(folder_name, frames[i].filename))
			img = cv2.imread(os.path.join(folder_name, frames[i].filename))
			state = zebra_crossing_detection.zebra_crossing(img, img.shape[0], img.shape[1])
		return state
	except Exception:
		return "<h1>ERROR</h1>"
if __name__ == "__main__":
	try:
		app.run(host='0.0.0.0',port=5000,debug=True,use_reloader=False) 
	except Exception:
		classification.session.close()
