import argparse
import Models
import queue
import cv2
import math
import numpy as np
import csv
import sys
from scipy.spatial import distance
import matplotlib.path as mpltPath
import IPython.display

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

#import tensorflow.keras.backend as K

from PIL import Image, ImageDraw

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"         # the ID of the GPU (view via nvidia-smi)

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type = str)
parser.add_argument("--output_file_path", type = str)

args = parser.parse_args()
input_video_path = args.input_video_path
output_file_path = args.output_file_path

# Initialize parameters and variables
save_weights_path = "weights/new_model_60.h5"
n_classes = 256

frame_step = 1
currentFrame = 0
last_frame_to_save = 0
to_save = False

# Background subtraction functions

def getDifferenceImage(cur_frame, last_frame):
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.GaussianBlur(cur_gray, (3, 3), 0)

    last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    last_gray = cv2.GaussianBlur(last_gray, (3, 3), 0)

    # Calculate absolute difference between current frame and reference frame
    # difference = cv2.absdiff(img1_gray, img2_gray)
    difference = cv2.subtract(cur_gray, last_gray)

    # Apply thresholding to eliminate noise
    thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = thresh.astype(np.uint8)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    return thresh

def getTwoEdgePoints(c, MA, ma, angle):
    angle = 90 - angle
    angle = math.radians(angle)
    d = (MA / 2) - (ma / 2)
    
    x1 = c[0] + (d * math.cos(angle))
    y1 = c[1] - (d * math.sin(angle)) # - , because of rectangle angle
    P1 = (x1, y1)

    x2 = c[0] + (-d * math.cos(angle))
    y2 = c[1] - (-d * math.sin(angle)) # - , because of rectangle angle
    P2 = (x2, y2)
    
    return P1, P2

def getMinDistPoint(points, p):
    distances = []
    
    for point in points:
        distance = math.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)
        distances.append(distance)
        
    minPoint = points[0]
    minDist = distances[0]
    
    for index, distance in enumerate(distances):
        if distance < minDist:
            minDist = distance
            minPoint = points[index]
    
    return minPoint, minDist

def getMaxDistPoint(P1, P2, p):
    distance1 = math.sqrt((p[0] - P1[0])**2 + (p[1] - P1[1])**2)
    distance2 = math.sqrt((p[0] - P2[0])**2 + (p[1] - P2[1])**2)
    
    if distance1 < distance2:
        return P2, distance2
    else:
        return P1, distance1

def getNearestTrajectory(point, trajectories):
    min_dist = math.inf
    min_point = None
    min_trajectory_num = None
    min_trajectory_point_num = None
    
    i = 0
    for trajectory in trajectories:
        num, trajectory_point, max_dist = trajectory
        
        dist = distance.euclidean(point, trajectory_point)
        if dist < max_dist and dist < min_dist:
            min_dist = dist
            min_point = point
            min_trajectory_num = num
            min_trajectory_point_num = i
        
        i = i + 1
        
    return [min_trajectory_num, min_trajectory_point_num, min_point]

def length(v):
    return np.sqrt(np.dot(v, v))

def angle(v1, v2):
    if length(v1) == 0 or length(v2) == 0:
        return 0
    
    # Angle in degrees
    return np.arccos(float("{:.4f}".format(np.dot(v1, v2) / (length(v1) * length(v2)))))

def calculateVertex(points):
    x = points.T[0]
    Y = points.T[1]
    
    X = np.array([x*x, x, np.ones(len(x))]).T
    
    if np.linalg.cond(X) < 1/sys.float_info.epsilon:
        X = np.linalg.inv(X)
    else:
        return None
    
    return X.dot(Y)

def getFixedParabola(parabola_points):
    p1, p2, p3 = parabola_points
    
    res_x = fixPoint([p1[0], p2[0], p3[0]])
    if res_x != None:
        p1_x, p2_x, p3_x = res_x
        #p1, p2, p3 = [p1_x, p1[1]], [p2_x, p2[1]], [p3_x, p3[1]]
        res_y = fixPoint([p1[1], p2[1], p3[1]])
        
        if res_y != None:
            p1_y, p2_y, p3_y = res_y
            
            return np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]])
    
    return None
            
def fixPoint(values):
    if values[0] == values[1]:
        if values[2] != values[1]:
            if values[2] > values[1]:
                values[1] += 0.1
            else:
                values[1] -= 0.1
        else:
            return None

    if values[1] == values[2]:
        if values[0] != values[1]:
            if values[1] > values[0]:
                values[2] += 0.1
            else:
                values[2] -= 0.1
        else:
            return None
        
    return values

def getNextPointBasedOnParabola(points):
    if getFixedParabola(points) is None:
        return None
    
    points = getFixedParabola(points)
    
    if calculateVertex(points) is None:
        return None
    
    a, b, c = calculateVertex(points)
    
    x = points[2][0] + (points[2][0] - points[1][0])
        
    # Parabola using calculated coefs
    y = a*x*x + b*x + c
    
    parab_expected_point = [round(x, 2), round(y, 2)]
    
    return parab_expected_point

def getDist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def getDistanceBetweenPoints(point1, point2):
    return distance.euclidean(point1, point2)
            
def saveCSVFile(csv_file, data, nextFrame):
    csv_columns = ['frame', 'ball_x', 'ball_y', 'key_event_x', 'key_event_y']
    file_exists = os.path.isfile(csv_file)
    
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = csv_columns)
            if not file_exists:
                writer.writeheader()
            
            for i in range(len(data)):
                item = data[i]
                frame = i + nextFrame
                
                if item is not None and item[1] is not None:
                    ball_x = float(item[1][0])
                    ball_y = float(item[1][1])
                else:
                    ball_x = None
                    ball_y = None
                    
                csv_data = {'frame': item[0], 'ball_x': ball_x, 'ball_y': ball_y, 'key_event_x': None, 'key_event_y': None}
                writer.writerow(csv_data)
    except IOError:
        print("I/O error")

def isTrajectoryWithinCourt(trajectory, court_boundaries):
    if len(trajectory) < 3:
        return False
    
    if trajectory[-1] is not None and trajectory[-2] is not None:
        # <= 120
        if getDistanceBetweenPoints(trajectory[-1], trajectory[-2]) > 5:
            court = mpltPath.Path(court_boundaries)
            print(court.contains_point(trajectory[-1]), court.contains_point(trajectory[-2]))
            if court.contains_point(trajectory[-1]) and court.contains_point(trajectory[-2]):
                return True
            
    return False

# Ball position detection functions
def calculateBlurredBallPosition(predicted_pos, contour, img1, img2):
    x, y = predicted_pos
    pred_X = x
    pred_Y = y
    
    ellipse = cv2.fitEllipse(contour)
    center, (MA, ma), angle = ellipse
    
    # Get background subtraction result (difference between previous and current images)
    prev_diff_img = getDifferenceImage(img2, img1)
    
    # Get rectangle and axis lengths for the contour
    rect = cv2.minAreaRect(contour)
    mjRA = max(rect[1][0], rect[1][1])
    mnRA = min(rect[1][0], rect[1][1])
    
    contours, _ = cv2.findContours(prev_diff_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    prevContourCenter = (x, y)
    
    minPrevDist = 60
    for c in contours:
        if (len(c) >= 5):
            prev_ellipse = cv2.fitEllipse(c)
            prev_center, (prev_MA, prev_ma), prev_angle = prev_ellipse
            
            dist = getDist(prev_center, center)
            if abs(prev_angle - angle) < 7 and abs(prev_MA - MA) < 2 and dist < minPrevDist:
                prevContourCenter = prev_center
                minPrevDist = dist
    
    # Find two edge points of the contour
    P1, P2 = getTwoEdgePoints(rect[0], mjRA, mnRA, angle)

    # Find correct edge point (based on previous ball position)
    # If previous ball position found and dist is small
    if minPrevDist < 60:
        point, _ = getMaxDistPoint(P1, P2, prevContourCenter)
        if cv2.pointPolygonTest(contour, point, True) > 0:
            pred_X, pred_Y = point
    else:
        minPoint, minDist = getMinDistPoint([P1, P2], (x, y))
                                
        if minDist <= 7:
            nextDist = cv2.pointPolygonTest(contour, minPoint, True)
            add = 1 # Decreasing Major axis length

            # Search for point until it is inside contour
            while nextDist <= 0:
                P1, P2 = getTwoEdgePoints(rect[0], mjRA - add, mnRA, angle)
                minPoint, minDist = getMinDistPoint([P1, P2], (x, y))

                nextDist = cv2.pointPolygonTest(contour, minPoint, True)
                
                add += 1
                if add > 5:
                    break

            if add <= 5:
                pred_X, pred_Y = minPoint
                
    
    return pred_X, pred_Y

def getBallPosition(predicted_pos, img1, img2):
    x, y = predicted_pos
    pred_X = x
    pred_Y = y
    
    # Get background subtraction result (difference between two current and previous images)
    diff_img = getDifferenceImage(img1, img2)    
    contours, _ = cv2.findContours(diff_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate all moving objects (after background subtraction)
    for c in contours:  
        # Check axis lengths
        M = cv2.moments(c)
        if (M["m00"] == 0):
            continue

        # Check if predicted ball object is inside the found contour object
        dist = cv2.pointPolygonTest(c, (x, y), True)
        if dist < 0:
            continue

        # 5 points polygon
        if (len(c) >= 5):
            # Get ellipse axis lengths
            ellipse = cv2.fitEllipse(c)
            center, (MA, ma), angle = ellipse

            # Find major and minor lengths
            mjA = max(MA, ma)
            mnA = min(MA, ma)

            # If the object is not a circle and has prolonged form (blurred)
            if mjA / mnA > 1.3:
                pred_X, pred_Y = calculateBlurredBallPosition((x, y), c, img1, img2)
            else:
                # Use ellipse center coordinates  
                pred_X, pred_Y = center

        # Less 5 points polygon (can not use ellipse)
        else:
            circle = cv2.minEnclosingCircle(c)
            circle_center = circle[0]

            # Use circle center coordinates                        
            pred_X, pred_Y = circle_center

        # Stop iteration when contour is found
        break
        
    return pred_X, pred_Y


# Get video fps and size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 60:
    frame_step = 2

# Width and height in TrackNet for prediction
width, height = 640, 360
img, img1, img2 = None, None, None

# Load TrackNet model
modelFN = Models.TrackNet.TrackNet
m = modelFN(n_classes, input_height = height, input_width = width)
m.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
m.load_weights(save_weights_path)

main_trajectory = []


#### Prepare first two frames ####
video.set(1, currentFrame); 
ret, img1 = video.read()
# Resize
img1 = cv2.resize(img1, ( width , height ))
img1 = img1.astype(np.float32)

main_trajectory.append([currentFrame, None])
currentFrame += frame_step

video.set(1, currentFrame);
ret, img = video.read()
# Resize
img = cv2.resize(img, ( width , height ))
img = img.astype(np.float32)

main_trajectory.append([currentFrame, None])

currentFrame += frame_step

#### Main loop ####
while(True):
    print(currentFrame)
    
    img2 = img1
    img1 = img

    video.set(1, currentFrame); 
    ret, img = video.read()

    # If no frames -> break
    if not ret: 
        break

    # Save initial image
    output_img = img
    
    # Process image
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)

    # Combine 3 images and predict
    X = np.concatenate((img, img1, img2), axis = 2)
    X = np.rollaxis(X, 2, 0)
    
    # Predict heatmap
    pr = m.predict(np.array([X]), batch_size=len(np.array([X])))[0]
    #K.clear_session()
    
    print("PREDICTED")

    # Reshape image
    pr = pr.reshape((height, width, n_classes)).argmax(axis = 2)
    pr = pr.astype(np.uint8) 
    heatmap = cv2.resize(pr, (output_width, output_height))
    
    # Threshold heatmap
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    
    #heatmap_math = heatmap.astype(np.uint8) 
    #heatmap_math = Image.fromarray(np.uint8(heatmap_math) , 'L')
    #imgplot = plt.imshow(heatmap_math)
    #plt.show()
    
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp = 1, minDist = 10, param2 = 2, minRadius = 2, maxRadius = 7)
    
    if circles is not None:
        # Predicted object coordinates
        x = int(circles[0][0][0])
        y = int(circles[0][0][1])

        # Calculate real ball position using Background subtraction
        x, y = getBallPosition((x, y), img, img1)

        main_trajectory.append([currentFrame, (x, y)])
    else:
        main_trajectory.append([currentFrame, None])

    # Next frame
    currentFrame += frame_step
        
    '''
    if currentFrame % 10 == 0:
        #to_save = True

    if to_save and len(main_trajectory) > 5:
        if (main_trajectory[-1] is None) and (main_trajectory[-2] is None) and (main_trajectory[-3] is None) and (main_trajectory[-4] is None) and (main_trajectory[-5] is None):
            if len(trajectories) == 0:
                # Download a csv file with raw ball positions data
                saveCSVFile(output_file_path, main_trajectory, last_frame_to_save)
                main_trajectory = []

                last_frame_to_save = currentFrame
                to_save = False
    '''

print("Finish")

# Download a csv file with raw ball positions data
saveCSVFile(output_file_path, main_trajectory, 0)