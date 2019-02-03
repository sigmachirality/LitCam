from imageai.Prediction.Custom import CustomImagePrediction
import os
execution_path = os.getcwd()
import math
import cv2
import numpy as np
import imutils

def detectShape(c):
    """
    Takes in an contour object and returns the shape type
    :param c: OpenCV contour of shape of interest
    :return: string name of shape
    """
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
 
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
    # compute the bounding box of the contour and use the
    # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
 
    # a square will have an aspect ratio that is approximately
    # equal to one, otherwise, the shape is a rectangle
        shape = "square" if (ar == 0.95 and ar <= 1.05) else "rectangle"
 
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
 
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
 
    # return the name of the shape
    return shape

def process_image(image):

    #Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    hsv = cv2.blur(hsv, (6,6))

    lower_grey = np.array([0, 6, 79])
    upper_grey = np.array([39, 51, 184])
    mask = cv2.inRange(hsv, lower_grey, upper_grey)

    lower_glare = np.array([0, 0, 175])
    upper_glare = np.array([30, 30, 285])
    glare = cv2.inRange(hsv, lower_glare, upper_glare)
    
    mask = cv2.bitwise_or(mask, glare)

    #Resize mask so that image can be processed faster
    #resized = imutils.resize(mask, width=300)
    #ratio = mask.shape[0]/ float(resized.shape[0])
    #TODO: implement image resizing for efficiency if it would actually help
    #resized = mask

    #Blur the resized image slightly, and threshold it
    #blurred = cv2.GaussianBlur(resized, (1, 1), 0)
    thresh = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    #Find contours in the thresholded image TODO: detect shape and only
    #draw ROI around squares (method for checking shape is above)


    #cnt_image = thresh.copy()
    cnt_image = cv2.convertScaleAbs(closing)
    cnt_image = cv2.bitwise_not(cnt_image)
    #cv2.imshow("meme", cv2.resize(cnt_image, (403,302)))
    #cv2.waitKey()

    cnts = cv2.findContours(cnt_image, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnts = cnts[0]
    #gray_image = cv2.convertScaleAbs(thresh)

    #rects = []
    images = []
    areas = []
    for c in cnts:
        a = cv2.contourArea(c)
        areas.append(a)
    for c in cnts:
        #break
        # print(type(c))
        # for i in c:
        #     print i
        #c = c.astype(int)
        #cv2.convertScaleAbs(c)
        a = cv2.contourArea(c)
        if (a > 12000):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            # if height is enough
            # create rectangle for bounding and draw it onto the image
            # commented out rectangle code because we're not using it...yet?
            #rect = (x, y, w, h)
            #rects.append(rect)
            #cv2.rectangle(image, (x - 3, y - 3), (x+w + 3, y+h + 3), (0, 0, 255), 1)
            #cv2.circle(image, (x + w / 2, y + h / 2), w / 2 + 40, (0, 0, 255), 2)
            images.append(image[y:(y + h),x:(x + w)])

    #Return object with bounding boxes
    return images

def classify_trash(images):
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, "model.h5"))
    prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
    prediction.loadModel(num_objects=6)
    trashTypes = set()
    for i in images:
        pre, prob = prediction.predictImage(i, input_type="array")
        trashTypes.add(pre[0])
        print(pre)
        print(prob)
    return trashTypes

test = cv2.imread(os.path.join(execution_path, 'test.jpg'))
tests = process_image(test)
print(classify_trash(tests))

