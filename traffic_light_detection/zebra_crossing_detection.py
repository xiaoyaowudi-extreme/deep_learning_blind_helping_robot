#===========================================#
#                                           #
#                                           #
#----------CROSSWALK RECOGNITION------------#
#-----------WRITTEN BY N.DALAL--------------#
#-----------------2017 (c)------------------#
#                                           #
#                                           #
#===========================================#

#Copyright by N. Dalal, 2017 (c)

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

import numpy as np
import cv2
import math
import scipy.misc
import PIL.Image
import statistics
import timeit
import glob
from sklearn import linear_model, datasets

#==========================#
#---------functions--------#
#==========================#

#get a line from a point and unit vectors
def lineCalc(vx, vy, x0, y0):
    scale = 10
    x1 = x0+scale*vx
    y1 = y0+scale*vy
    m = (y1-y0)/(x1-x0)
    b = y1-m*x1
    return m,b

#the angle at the vanishing point
def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    #print(len1)
    #print(len2)
    a=math.acos(inner_product/(len1*len2))
    return a*180/math.pi 

#vanishing point - cramer's rule
def lineIntersect(m1,b1, m2,b2) : 
    #a1*x+b1*y=c1
    #a2*x+b2*y=c2
    #convert to cramer's system
    a_1 = -m1 
    b_1 = 1
    c_1 = b1

    a_2 = -m2
    b_2 = 1
    c_2 = b2

    d = a_1*b_2 - a_2*b_1 #determinant
    dx = c_1*b_2 - c_2*b_1
    dy = a_1*c_2 - a_2*c_1

    intersectionX = dx/d
    intersectionY = dy/d
    return intersectionX,intersectionY

#process a frame
def process(im,W,H):
    start = timeit.timeit() #start timer

    #initialize some variables
    x = W
    y = H

    radius = 250 #px
    thresh = 170 
    bw_width = 170 

    bxLeft = []
    byLeft = []
    bxbyLeftArray = []
    bxbyRightArray = []
    bxRight = []
    byRight = []
    boundedLeft = []
    boundedRight = []
    
    #1. filter the white color
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(im,lower,upper)

    #2. erode the frame
    erodeSize = int(y / 30)
    #print(y)
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize,1))
    erode = cv2.erode(mask, erodeStructure, (-1, -1))

    #3. find contours and  draw the green lines on the white strips
    _ , contours,hierarchy = cv2.findContours(erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
 
    for i in contours:
        bx,by,bw,bh = cv2.boundingRect(i)
        if (bw > bw_width):
            bxRight.append(bx+bw) #right line
            byRight.append(by) #right line
            bxLeft.append(bx) #left line
            byLeft.append(by) #left line
            bxbyLeftArray.append([bx,by]) #x,y for the left line
            bxbyRightArray.append([bx+bw,by]) # x,y for the left line

    #calculate median average for each line
    medianR = np.median(bxbyRightArray, axis=0)
    medianL = np.median(bxbyLeftArray, axis=0)

    bxbyLeftArray = np.asarray(bxbyLeftArray)
    bxbyRightArray = np.asarray(bxbyRightArray)
     
    #4. are the points bounded within the median circle?
    for i in bxbyLeftArray:
        if (((medianL[0] - i[0])**2 + (medianL[1] - i[1])**2) < radius**2) == True:
            boundedLeft.append(i)

    boundedLeft = np.asarray(boundedLeft)

    for i in bxbyRightArray:
        if (((medianR[0] - i[0])**2 + (medianR[1] - i[1])**2) < radius**2) == True:
            boundedRight.append(i)

    boundedRight = np.asarray(boundedRight)

    #5. RANSAC Algorithm

    #select the points enclosed within the circle (from the last part)
    bxLeft = np.asarray(boundedLeft[:,0])
    byLeft =  np.asarray(boundedLeft[:,1]) 
    bxRight = np.asarray(boundedRight[:,0]) 
    byRight = np.asarray(boundedRight[:,1])

    #transpose x of the right and the left line
    bxLeftT = np.array([bxLeft]).transpose()
    bxRightT = np.array([bxRight]).transpose()

    #run ransac for LEFT
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    ransacX = model_ransac.fit(bxLeftT, byLeft)
    inlier_maskL = model_ransac.inlier_mask_ #right mask

    #run ransac for RIGHT
    ransacY = model_ransac.fit(bxRightT, byRight)
    inlier_maskR = model_ransac.inlier_mask_ #left mask


    #6. Calcuate the intersection point of the bounding lines
    #unit vector + a point on each line
    vx, vy, x0, y0 = cv2.fitLine(boundedLeft[inlier_maskL],cv2.DIST_L2,0,0.01,0.01) 
    vx_R, vy_R, x0_R, y0_R = cv2.fitLine(boundedRight[inlier_maskR],cv2.DIST_L2,0,0.01,0.01)

    #get m*x+b
    m_L,b_L=lineCalc(vx, vy, x0, y0)
    m_R,b_R=lineCalc(vx_R, vy_R, x0_R, y0_R)

    #calculate intersention 
    intersectionX,intersectionY = lineIntersect(m_R,b_R,m_L,b_L)

   
    #8. calculating the direction vector  
    POVx = W/2 #camera POV - center of the screen
    POVy = H/2 # camera POV - center of the screen

    Dx = -int(intersectionX-POVx) #regular x,y axis coordinates
    Dy = -int(intersectionY-POVy) #regular x,y axis coordinates

    #focal length in pixels = (image width in pixels) * (focal length in mm) / (CCD width in mm)
    focalpx = int(W * 4.26 / 6.604) #all in mm


    end = timeit.timeit() #STOP TIMER
    time_ = end - start

    #print('DELTA (x,y from POV):' + str(Dx) + ',' + str(Dy))
    return im,Dx,Dy
    
#=============================#
#---------MAIN PROGRAM--------#
#=============================#
#initialization

Dx = []
Dy = []
after =0
DxAve =0
Dxold =0
DyAve =0
Dyold =0
i = 0

def zebra_crossing(img, _H, _W):
    global Dx,Dy,after,DxAve,Dxold,DyAve,Dyold,i
    #print([Dx,Dy,after,DxAve,Dxold,DyAve,Dyold,i])
    state = ""
    cv2.circle(img,(int(_W/2),int(_H/2)),5,(0,0,255),8)
    H = _H
    W = _W
    processedFrame,dx,dy = process(img,_W,_H)

    if (i < 3):
        Dx.append(dx)
        Dy.append(dy)
        i=i+1

    else:
        DxAve = sum(Dx)/len(Dx)
        DyAve = sum(Dy)/len(Dy)
        del Dx[:]
        del Dy[:]
        i=0
    # Dx.append(dx)
    # Dy.append(dy)
    # i=i+1
    # DxAve = sum(Dx)/len(Dx)
    # DyAve = sum(Dy)/len(Dy)
    # del Dx[:]
    # del Dy[:]
    # i=0

    if (DyAve > 30) and (abs(DxAve) < 300):        
        #check if the vanishing point and the next vanishing point aren't too far from each other 
        if (((DxAve - Dxold)**2 + (DyAve - Dyold)**2) < 150**2) == True:  ##distance 150 px max 
            
            #walking directions
            if abs(DxAve) < 80 and DyAve > 100 and abs(Dxold-DxAve) < 20:
                state = 'Straight'

            elif DxAve > 80 and DyAve > 100 and abs(Dxold-DxAve) < 20:
                state = 'Right'

            elif DxAve < 80 and DyAve > 100 and abs(Dxold-DxAve) < 20:
                state = 'Left'         

        Dxold = DxAve
        Dyold = DyAve
            
        #walking directions
        # if abs(DxAve) < 80 and DyAve > 100 and abs(Dxold-DxAve) < 20:
        #     state = 'Straight'

        # elif DxAve > 80 and DyAve > 100 and abs(Dxold-DxAve) < 20:
        #     state = 'Right'

        # elif DxAve < 80 and DyAve > 100 and abs(Dxold-DxAve) < 20:
        #     state = 'Left'
    return state
