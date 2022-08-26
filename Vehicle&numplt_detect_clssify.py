import os
import cv2
import time
import base64
import string
import random
import imutils
import datetime
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES,hex_to_rgb


model_path = 'model/model.tflite'
classes = ['license_plate']

DETECTION_THRESHOLD = 0.5

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results

def run_odt_and_draw_results(image_path, interpreter, threshold=0.9):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  print("Number detect :)")
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    detected = cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
    numplt_crop = detected[ymin:ymax, xmin:xmax]
    randm_name = (''.join(random.choices(string.ascii_letters, k=3)))
    crop_name = randm_name+'.jpg'
    path = "numplt_crop/"+crop_name
    cv2.imwrite(path, numplt_crop)
    
    with open(path, "rb") as img_file:
        basifo = base64.b64encode(img_file.read())
    try:
        r = requests.post('http://192.168.1.79:8081/input_image', data ={'file': basifo, 'crop': crop_name})
        print(r)
    except Exception as e:
        print("Exception post request::::", e)
    
  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8

# COLOR RECOGNITION OF VEHICLE]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

def most_common_used_color(img):
	width, height = img.size
	r_total = 0
	g_total = 0
	b_total = 0
 
	count = 0
	for x in range(0, width):
		for y in range(0, height):
			r, g, b = img.getpixel((x, y))
			r_total += r
			g_total += g
			b_total += b
			count += 1
	return (round(r_total/count), round(g_total/count), round(b_total/count))

def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'


# VEHICLE DETECTION]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

classname = []
list_of_vehicles = ["car","motorbike","bicycle","bus"]

labelsPath = 'yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"
print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

video = cv2.VideoCapture('20220729115945.mp4')                                               # INPUT VIDEO ################################ 

(W, H) = (None, None)
print(video)
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = 200
	# print("[INFO] {} total frames in video".format(total))
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
	(grabbed, frame) = video.read()
	if not grabbed:
		break
	print("FRAME grabbed", grabbed)
	
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	try:
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		boxes = []
		confidences = []
		classIDs = []
	   
		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				
				if confidence > 0.8:
					if LABELS[classID] in list_of_vehicles:
						print('Label class::::::::::::::',LABELS[classID])
							
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						classname.append(LABELS[classID])
					
		# apply non-maxima suppression to suppress weak, overlapping bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.5)
		
		# ensure at least one detection exists
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
  
				color = [int(c) for c in COLORS[classIDs[i]]]
				bob = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				
				vehicle_crop = bob[y:y+h, x:x+w]
                
				rand_name = ''.join(random.choices(string.ascii_letters, k=3))
				vehicle_img = 'vehicles/'+rand_name+'.jpg'
				vehicle = cv2.imwrite(vehicle_img, vehicle_crop)

				img = Image.open(r"vehicles/"+rand_name+".jpg")
				img = img.convert('RGB')
				common_color = most_common_used_color(img)
				v_color = convert_rgb_to_names(common_color)
				print('color of vehicle=============',v_color)
				print('vehicle...........', vehicle)
				try:
					# Run inference and draw detection result on the local copy of the original file
					detection_result_image = run_odt_and_draw_results(
						vehicle_img,
						interpreter,
						threshold=DETECTION_THRESHOLD
					)
				except Exception as e:
					print('not detect number plate from vehicle crop__________\n',e)
    
			if total > 0:
				elap = (end - start)
				print("[INFO] single frame took {:.4f} seconds".format(elap))
				print("[INFO] estimated total time to finish: {:.4f}".format(
					elap * total))
		else:
			randm_name = (''.join(random.choices(string.ascii_letters, k=3)))
			current_time = datetime.datetime.now()
			cv2.imwrite("without_numplt_detect/"+current_time+'_'+randm_name+'.jpg',vehicle_crop)
	except Exception as e:
          print('printing e-----',e)

# release the file pointers
print("[INFO] cleaning up...")
