#import the necessary packages
from collections import deque	   #to store past N points of track object 
import numpy as np			       #to store data in matrix from 
import argparse			           #to give easy video path in command line interface
import cv2
#1********************************************** 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()                          #to hold all information necessary to parse command line into python dataype(ap).
ap.add_argument("-v", "--video")                        #it informs argparser to take string on command line n turns thm into object
ap.add_argument("-b", "--buffer", type=int, default=32)		#to store max size of dequeue we use buffer 
args = vars(ap.parse_args())                                   # the info given by add arg is store n used in parse arg
# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = (29, 86, 6)                                #hsv is used for specific clour extraction
greenUpper = (64, 255, 255)
 
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])     #list of tracked points
counter = 0
(dX, dY) = (0, 0)
direction = ""
 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)                    #if we use any external device to capture video we use arg as 1.
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])
	#1ends********************************************************
	#2 strts
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()                
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
 
	# convert it to the HSV
	# color space
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)		#hsv to binary (green)thresholded image
	mask = cv2.erode(mask, None)                            #it erodes overlapping boundaries between two surrounding pixels
	mask = cv2.dilate(mask, None)
 
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	
	# Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

	#For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
	#findContours function modifies the source image. So if you want source image even after finding contours, already store it to some other variables.
	#In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.
	# arguments are first one is source image, second is contour retrieval mode, third is contour approximation method.

	# cv::RETR_EXTERNAL = 0, retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours. 
	# cv::CHAIN_APPROX_SIMPLE = 2, compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points. 

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	#2end************************************************************************************************************************8
		# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)              #to find max contour of object
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			pts.appendleft(center)
		# loop over the set of tracked points
		#*****************************************************************************************************************************88888

	# Parameters for (np.arange)
	#start : number, optional Start of interval.  The interval includes this value.  The default start value is 0.
	#stop : number, End of interval.  The interval does not include this value.
	#step : number, optional Spacing between values.  For any output `out`, this is the distance 
		# between two adjacent values, ``out[i+1] - out[i]``.  The default
    		# step size is 1.  If `step` is specified, `start` must also be given.
	#dtype : dtype, The type of the output array.  If `dtype` is not given, infer the data type from the other input arguments.
 
	#Returns
	#out : ndarray, Array of evenly spaced values.

	for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
 
		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 1 and pts[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]			#computes delta x and y of current frame and  end of buffer to handle directional movement
			(dirX, dirY) = ("", "")
 
			# ensure there is significant movement in the
			# x-direction
			# check the magnitude of the x-delta to see if there is a  significant difference in direction along the x-axis. In 			this case, if there is more than 20 pixel difference between the x-coordinates, we need to figure out in which 			direction the object is moving. If the sign of dX  is positive, then we know the object is moving to the right 			(east). Otherwise, if the sign of dX is negative, then we are moving to the left (west).

			#np.abs -- Calculate the absolute value element-wise.
			#Parameters: x : array_like Input array.
			#Returns: absolute : ndarray, An ndarray containing the absolute value of each element in x. For complex input, a + ib, the absolute value is \sqrt{ a^2 + b^2 }.

			if np.abs(dX) > 20:
				dirX = "East" if np.sign(dX) == 1 else "West" 	#if dx is positive then right(east) else left(west)
								#np.sign - Returns an element-wise indication of the sign of a number.
 
			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 20:
				dirY = "North" if np.sign(dY) == 1 else "South"
 
			# handle when both directions are non-empty
			if dirX != "" and dirY != "":              #if dy positive then north else south
				direction = "{}-{}".format(dirY, dirX)
 
			# otherwise, only one direction is non-empty
			else:
				direction = dirX if dirX != "" else dirY		#to move diagonally
				#**************************************************************************************************************************888
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
	# show the movement deltas and the direction of movement on
	# the frame
	cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 0, 255), 3)
	cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
 
	# show the frame to our screen and increment the frame counter
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	counter += 1
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
