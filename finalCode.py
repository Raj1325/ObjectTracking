import cv2
import numpy as np
import math
import serial
from time import sleep

arduino = serial.Serial('COM4', 115200, timeout=.1)
# Region-of-interest vertices 
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 1.0  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 1.0  # ditto for top edge of trapezoid
trap_height = 1.0  # height of the trapezoid expressed as percentage of image height


def region_of_interest(img, vertices): 
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
                channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,) * channel_count
        else:
                ignore_mask_color = 255
                
        #filling pixels inside the polygon defined by "vertices" with the fill color	
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    

width_frame = 720
height_frame = 480
threshold = 80
camera = cv2.VideoCapture(0)
while(True):
        
        (grabbed, img) = camera.read()
        #cv2.imwrite('rdf.jpg',img)
        resized_image = cv2.resize(img, (width_frame, height_frame))
        #cv2.imshow('original image',resized_image)
        hsi = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HLS)
        #cv2.imshow('hsi image',hsi)

        Z = hsi.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((hsi.shape))

        #cv2.imshow('res2',res2)


###############
        blur = cv2.bilateralFilter(res2,9,75,75)

        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        img = resized_image.copy()

        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        
# Create masked edges using trapezoid-shaped region-of-interest
        imshape = resized_image.shape
        vertices = np.array([[\
                ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
                ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
                (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
                (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
                , dtype=np.int32)
        masked_edges = region_of_interest(edges, vertices)
        mask = cv2.inRange(masked_img, redLower, redUpper)		#hsv to binary (green)thresholded image
        mask = cv2.erode(mask, None)                            #it erodes overlapping boundaries between two surrounding pixels
        mask = cv2.dilate(mask, None)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
        if(maxVal > 200):
                arduino.write(b'x')
                print "RedSignal"
                sigFlag = 1
        else:
                arduino.write(b'f')
                print "GreenSig"
                sigFlag = 0

        left=0
        right=0
        leftu=0
        rightu=0
        leftm = 0
        rightm = 0
        counterleft = 0
        counterright=0
        lines = cv2.HoughLines(masked_edges, 1, np.pi/180, 70)
        try:
                        range = lines.shape[0]
        except AttributeError:
                        range = 0

        for i in xrange(range):
                        for rho, theta in lines[i]:
                                        if rho > 0 and (np.pi*1/10 < theta < np.pi*4/10):
                                                        if(counterleft<3):
                                                                a = np.cos(theta)
                                                                b = np.sin(theta)
                                                                x0 = a * rho
                                                                y0 = b * rho
                                                                x1 = int(x0 + 1000 * (-b))
                                                                y1 = int(y0 + 1000 * (a))
                                                                x2 = int(x0 - 1000 * (-b))
                                                                y2 = int(y0 - 1000 * (a))
                                                                print 'left'
                                                                print x1
                                                                print y1
                                                                print x2
                                                                print y2

                                                                A = (x1, y1)
                                                                B = (x2, y2)
                                                                #print(angle_between(A, B))
                                
                                                                dx,dy = x2-x1,y2-y1
                                                                #angle = math.atan(float(dx)/float(dy))
                                                                #angle *= 180/math.pi
                                                                #print angle
                                                                ss = (float(dy)/float(dx))
                                                                #print float(ss)
                                                                left = ((height_frame-y2)/float(ss))+x2
                                                                #print left
                                                                leftu = ((0-y2)/float(ss))+x2
                                                                leftm = ((230-y2)/float(ss))+x2
                                                                #cv2.line(img, (int(left), height_frame), (355, 10), (0, 255, 0))
                                                                #cv2.line(img, (359, height_frame), (720, 0), (0, 255, 0))
                                                                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))
                                                                counterleft = counterleft +1
                        

                                        if rho < 0 and (np.pi*7/10 < theta < np.pi*9/10):
                                                        if(counterright<3):
                                                                a = np.cos(theta)
                                                                b = np.sin(theta)
                                                                x0 = a * rho
                                                                y0 = b * rho
                                                                x1 = int(x0 + 1000 * (-b))
                                                                y1 = int(y0 + 1000 * (a))
                                                                x2 = int(x0 - 1000 * (-b)) 
                                                                y2 = int(y0 - 1000 * (a))
                                                                print x1
                                                                print y1
                                                                print x2
                                                                print y2
                                                                dx,dy = x2-x1,y2-y1
                                                                ss = (float(dy)/float(dx))
                                                                #print float(ss)
                                                                right = ((height_frame-y2)/float(ss))+x2
                                                                #print right
                                                                rightu = ((0-y2)/float(ss))+x2
                                                                rightm = ((230-y2)/float(ss))+x2
                                                                #cv2.line(img, (int(right), 350), (355, 10), (0, 255, 0))
                                                                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))
                                                                counterright = counterright +1
                                        if((counterleft >=3) and (counterright >=3)):
                                                break
                                        

        print left
        print right
        mid = (right+left)/2
        midu = (rightu+leftu)/2
        cv2.line(img, (int(mid), height_frame), (int(midu), 0), (0, 255, 0))

        mx,my = (int(mid)+int(midu))/2 , (height_frame+0)/2
        print mx
        print my

        lmx , lmy = width_frame/2 , height_frame/2
        
        if mx > lmx:
                if abs(mx-lmx) > threshold:
                        print "go right"
                        print "Turning motor on"
                        arduino.write(b'r')
                else:
                        print "go straight"
                        arduino.write(b'f')
        elif mx < lmx:
                if abs(mx-lmx) > threshold:
                        print "go left"
                        arduino.write(b'l')
                else:
                        print "go straight"
                        arduino.write(b'f')
                
        elif mx == lmx:
                print "go straight"
                arduino.write(b'f')
        

        cv2.line(img, (int(leftm), 230), (int(rightm), 230), (0, 255, 0))
        cv2.line(img, (int(leftm), 230), (355, height_frame), (0, 255, 0))
        cv2.line(img, (int(rightm), 230), (355, height_frame), (0, 255, 0))
        #cv2.imshow('djs.jpg',img)
        
#dist = sqrt( (int(leftm) - 355)**2 + (230 - 350)**2 )
#print dist
        distl = math.hypot(int(leftm) - 355, 230 - height_frame)
        print distl
        distr = math.hypot(int(rightm) - 355, 230 - height_frame)
        print distr
"""
        if distl > distr:
                if abs(mx-lmx) > threshold:
                        print "go left"
                        print "Turning motor on"
                        arduino.write(b'l')
                else:
                        print "go straight"
                        arduino.write(b'f')

        elif distl < distr:
                if abs(mx-lmx) > threshold:
                        print "go right"
                        arduino.write(b'r')
                else:
                        print "go straight"
                        arduino.write(b'f')
        elif distl == distr:
                print "go straight"
                arduino.write(b'f')
        
"""

camera.close()
arduino.write(b's')
cv2.destroyAllWindows()
