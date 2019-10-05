import requests
import os

import config
import cv2
cap = cv2.VideoCapture(0)

def get_zebra_crossing_status():
	global cap
	frames = {}
	for i in ramge(6):
		_, frame = cap.read()
		cv2.imwrite("wait_for_uploads/frame_" + str(i+1) + ".jpg", frame)
		frames['frame_' + str(i+1)] = ( "frame_" + str(i+1) + ".jpg", open("wait_for_uploads/frame_" + str(i+1) + ".jpg", "rb"), "image/jpg")
	return frames

try:
	while True:
		_, frame = cap.read()
		cv2.imwrite("wait_for_uploads/frame.jpg", frame)
		text = requests.post(config.url, {'file' : ("frame.jpg", open("wait_for_uploads/frame.jpg", "rb"), "image/jpg" ) } )
		if 'No' in text.text:
			continue
		if 'Red' in text.text:
			os.system( 'espeak -vzh -g 2 -s 155 -a 200 "已经检测到红灯, 禁止通行"')
			continue
		if "ERROR" in text.text:
			continue
		while True:
			info = requests.post(get_zebra_crossing_status()).text
			if info == "straight":
				os.system( 'espeak -vzh -g 2 -s 155 -a 200 "请通行"')
				break
			elif info == "Right":
				os.system( 'espeak -vzh -g 2 -s 155 -a 200 "请向右转一些"')
			elif info == "Left":
				os.system( 'espeak -vzh -g 2 -s 155 -a 200 "请向左转一些"')
except Exception:
	pass
