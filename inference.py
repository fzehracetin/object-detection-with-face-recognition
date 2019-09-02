import os
import sys

if( len(sys.argv) != 2 ):
	print( "You must define 1 mode: (webcam, video, image)")
	exit()


DETECTION_MODE = sys.argv[1] #webcam , video or image

if( DETECTION_MODE not in ["webcam", "video", "image"] ):
	print( "Mode must be one of these: (webcam, video, image) ")
	exit()

import cv2
import numpy as np
import tensorflow as tf

from scipy import ndimage

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


def detectOnVideo():
	VIDEO_NAME = 'test1.mp4'

	# Path to video
	PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

	# Open video file
	video = cv2.VideoCapture(PATH_TO_VIDEO)

	# Open output video file
	WRITTENVIDEO_PATH = CWD_PATH + "/output.avi"

	frame_width = int(video.get(3))
	frame_height = int(video.get(4))

	new_width = int(frame_width/2)
	new_height = int(frame_height/2)

	print( "---------",frame_width,",",frame_height,"------FrameCount: ", video.get(7))

	videoWriter = cv2.VideoWriter(WRITTENVIDEO_PATH, cv2.VideoWriter_fourcc('M','J', 'P', 'G'), 
		10,(new_height,new_width))

	fcount=0
	while(video.isOpened()):

		ret, frame = video.read()		
		if ret == True:

			frame = ndimage.rotate(frame, 270)
			frame = cv2.resize(frame, (new_height, new_width))
			
			frame_expanded = np.expand_dims(frame, axis=0)
		    # Perform the actual detection by running the model with the image as input
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: frame_expanded})
		   
		    # Draw the results of the detection (aka 'visualize the results')
			vis_util.visualize_boxes_and_labels_on_image_array(
				frame,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8,
				min_score_thresh=0.60) # 0.60 is default value

			videoWriter.write(frame)

			if( fcount % 20 == 0 ):
				print(fcount)

			fcount = fcount + 1
	    	# Press 'q' to quit
			#if cv2.waitKey(1) == ord('q'):
			#	break

		else:
			break


	print("Successfull")
	# Clean up
	video.release()
	videoWriter.release()
	cv2.destroyAllWindows()

def detectOnImage():
	IMAGE_NAME = 'test1.png'

	# Path to image
	PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
	# Load image using OpenCV and
	# expand image dimensions to have shape: [1, None, None, 3]
	# i.e. a single-column array, where each item in the column has the pixel RGB value
	image = cv2.imread(PATH_TO_IMAGE)
	image_expanded = np.expand_dims(image, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
	    [detection_boxes, detection_scores, detection_classes, num_detections],
	    feed_dict={image_tensor: image_expanded})

	# Draw the results of the detection (aka 'visulaize the results')

	print(boxes.shape, scores.shape, classes.shape, num.shape)

	print(boxes, scores, classes)
	#print boxes[0][:5] , "--", scores[0][:5], "--", classes[0][:5], "--", num[0]
	#print classes[0][40:80]

	vis_util.visualize_boxes_and_labels_on_image_array(
	    image,
	    np.squeeze(boxes),
	    np.squeeze(classes).astype(np.int32),
	    np.squeeze(scores),
	    category_index,
	    use_normalized_coordinates=True,
	    line_thickness=8,
	    min_score_thresh=0.60) # default is 0.6

	# All the results have been drawn on image. Now display the image.
	cv2.imshow('Object detector', image)

	# Press any key to close the image
	cv2.waitKey(0)

	# Clean up
	cv2.destroyAllWindows()

def detectOnWebcam():
	# Initialize webcam feed
	video = cv2.VideoCapture("http://192.168.16.195:8080/stream.wmv")
	ret = video.set(3,1280)
	ret = video.set(4,720)

	while(True):

	    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
	    # i.e. a single-column array, where each item in the column has the pixel RGB value
	    ret, frame = video.read()
	    frame_expanded = np.expand_dims(frame, axis=0)

	    # Perform the actual detection by running the model with the image as input
	    (boxes, scores, classes, num) = sess.run(
	        [detection_boxes, detection_scores, detection_classes, num_detections],
	        feed_dict={image_tensor: frame_expanded})

	    # Draw the results of the detection (aka 'visulaize the results')
	    vis_util.visualize_boxes_and_labels_on_image_array(
	        frame,
	        np.squeeze(boxes),
	        np.squeeze(classes).astype(np.int32),
	        np.squeeze(scores),
	        category_index,
	        use_normalized_coordinates=True,
	        line_thickness=8,
	        min_score_thresh=0.60)

	    # All the results have been drawn on the frame, so it's time to display it.
	    cv2.imshow('Object detector', frame)

	    # Press 'q' to quit
	    if cv2.waitKey(1) == ord('q'):
	        break

	# Clean up
	video.release()
	cv2.destroyAllWindows()



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 9

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

if( DETECTION_MODE == "video"):
	detectOnVideo()
elif( DETECTION_MODE == "image"):
	detectOnImage()
else:
	detectOnWebcam()


