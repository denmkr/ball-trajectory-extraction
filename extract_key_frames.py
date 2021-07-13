import argparse
import numpy as np
from scipy.spatial import distance
import math
import csv

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_file_path", type = str)
parser.add_argument("--output_file_path", type = str)

args = parser.parse_args()
input_file_path = args.input_file_path
output_file_path = args.output_file_path


def length(v):
    return np.sqrt(np.dot(v, v))

def angle(v1, v2):
    if length(v1) == 0 or length(v2) == 0:
        return 0
    
    # Angle in degrees
    return math.degrees(np.arccos(np.dot(v1, v2) / (length(v1) * length(v2))))

def getDistanceBetweenPoints(point1, point2):
    return distance.euclidean(point1, point2)

def isTrajectoryChanged(points, point, step):
    a, b, c = calculateVertex(points)

    x = points[2][0] + (((points[2][0] - points[1][0]) / 2) * step)
    # Parabola using calculated coefs
    y = a*x*x + b*x + c
    
    parab_expected_point = [x, y]
    
    dist = getDistanceBetweenPoints(point, parab_expected_point)
    
    change = (points[2] - points[1]) / 2
    speed = np.sqrt((change[0] * change[0]) + (change[1] * change[1]))
    
    res = dist / speed
    
    return res

def getNextPositionBasedOnParabola(parabola_points):
    a, b, c = calculateVertex(parabola_points)

    x = parabola_points[2][0] + ((parabola_points[2][0] - parabola_points[1][0]) / 2)
    # Parabola using calculated coefs
    y = a*x*x + b*x + c
    
    return [x, y]

def pointsValuesEqual(points):
    if (points.T[0][0] == points.T[0][1] or points.T[0][0] == points.T[0][2] or points.T[0][1] == points.T[0][2] 
       or points.T[1][0] == points.T[1][1] or points.T[1][0] == points.T[1][2] or points.T[1][1] == points.T[1][2]):
        return True
    
    return False

def calculateLineIntersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return line2[0][0], line2[0][1]
        # raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    return x, y

def calculateVertex(points):
    x = points.T[0]
    Y = points.T[1]
    
    X = np.array([x*x, x, np.ones(len(x))]).T
    
    X = np.linalg.inv(X)
    
    return X.dot(Y)

def getDataFromFile(filename):
    points = []
    
    with open(filename, newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
        
        # Because 1 and 2 images are not considered in tracknet detection
        #points.append(None)
        #points.append(None)
        
        for i, row in enumerate(reader):
            print(row)
            # Skip first two frames and header
            if i == 0:
                continue
                
            if row[1] is not '': 
                x = float(row[1])
                y = float(row[2])
                points.append((x, y))
            else:
                points.append(None)
            
    return np.array(points)

def saveCSVFile(csv_file, dict_data):
    csv_columns = ['frame', 'ball_x', 'ball_y', 'key_event_x', 'key_event_y']
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                if data['ball_position'] is None:
                    data['ball_position'] = [None, None]
                if data['key_position'] is None:
                    data['key_position'] = [None, None]
                    
                csv_data = {'frame': data['frame'], 'ball_x': data['ball_position'][0], 'ball_y': data['ball_position'][1], 'key_event_x': data['key_position'][0], 'key_event_y': data['key_position'][1]}
                writer.writerow(csv_data)
    except IOError:
        print("I/O error")

def checkTrajectory(parabola_points, point1, point2, point3):
    res1 = isTrajectoryChanged(parabola_points, point1, 1)
    res2 = isTrajectoryChanged(parabola_points, point2, 2)
    res3 = isTrajectoryChanged(parabola_points, point3, 3)
    
    if (res1 > 0.5) and (res1 < 999) and (res2 > res1) and (res3 > res2):
        return [True, [res1, res2, res3]]
    
    return [False, [res1, res2, res3]]

def isParabolaBuild(parabola_points):
    if None in parabola_points:
        return False
    
    # To less distance -> Serve
    dist = getDistanceBetweenPoints(parabola_points[0], parabola_points[2])
    if dist < 25:
        return False
        
    # Parabola can not be built
    if pointsValuesEqual(parabola_points): 
        return False
    
    return True

def getTrajectoryChangeFrame(frame_number, j, traj_change_position, all_points):
    # Calculate frame at which the trajectory change happened (if there are None position) 
    skip_count = 0
    j = j + 1
    
    # For previous
    new_pos_prev = None
    if all_points[j-2] is not None:
        new_pos_prev = all_points[j-2]
    else:
        vector = [all_points[j+1][0] - all_points[j][0], all_points[j+1][1] - all_points[j][1]]
        if all_points[j-1] is not None:
            new_pos_prev = [all_points[j-1][0] - vector[0] * (skip_count + 1), all_points[j-1][1] - vector[1] * (skip_count + 1)]
        else:
            new_pos_prev = [all_points[j][0] - vector[0] * (skip_count + 1) * 2, all_points[j][1] - vector[1] * (skip_count + 1) * 2]
        
    new_dist_prev = getDistanceBetweenPoints(traj_change_position, new_pos_prev)

    # For next
    if all_points[j] is not None:
        new_pos_next = all_points[j]
    else:
        vector = [all_points[j-1][0] - all_points[j-2][0], all_points[j-1][1] - all_points[j-2][1]]
        new_pos_next = [all_points[j-1][0] - vector[0] * (skip_count + 1), all_points[j-1][1] - vector[1] * (skip_count + 1)]
        
    new_dist_next = getDistanceBetweenPoints(traj_change_position, new_pos_next)

    old_dist = getDistanceBetweenPoints(traj_change_position, all_points[j-1])

    if new_dist_next <= new_dist_prev:
        if new_dist_next < old_dist:
            new_pos_next = all_points[j+1]
            new_dist_next = getDistanceBetweenPoints(traj_change_position, new_pos_next)
            skip_count += 1

            if new_dist_next < old_dist:
                skip_count += 1
    else:
        while new_dist_prev < old_dist:
            old_dist = new_dist_prev
            skip_count -= 1

            new_pos_prev = None
            if all_points[j-2+skip_count] is not None:
                new_pos_prev = all_points[j-2+skip_count]
            else:
                vector = [all_points[j+1][0] - all_points[j][0], all_points[j+1][1] - all_points[j][1]]
                new_pos_prev = [all_points[j-1][0] - vector[0] * (skip_count * (-1) + 1), all_points[j-1][1] - vector[1] * (skip_count * (-1) + 1)]

            new_dist_prev = getDistanceBetweenPoints(traj_change_position, new_pos_prev)
            
    return frame_number + skip_count


all_points = getDataFromFile(input_file_path)

none_points = []

# Replace None and None, None by values based on a parabola
none_count = 0
for i in range(5, len(all_points)):
    if all_points[i] is None:
        if none_count < 2:
            # Check if parabola can be built
            parabola_points = np.array([all_points[i-5], all_points[i-3], all_points[i-1]])
            if isParabolaBuild(parabola_points):
                x, y = getNextPositionBasedOnParabola(parabola_points)
                if x >= 0 and y >= 0:
                    all_points[i] = (round(x, 1), round(y, 1))
                    none_points.append(i)
            
            none_count += 1
    else:
        none_count = 0


######


next_i = 5
output = []
key_frames = []
res_count = 0

output.append({'frame': 0, 'ball_position': all_points[0], 'key_position': None})
output.append({'frame': 1, 'ball_position': all_points[1], 'key_position': None})
output.append({'frame': 2, 'ball_position': all_points[2], 'key_position': None})
output.append({'frame': 3, 'ball_position': all_points[3], 'key_position': None})
output.append({'frame': 4, 'ball_position': all_points[4], 'key_position': None})

for i in range(5, len(all_points) - 2):
    
    # Save frame into list
    traj_change_position = None
    frame_number = i

    position = {'frame': frame_number, 'ball_position': all_points[i], 'key_position': traj_change_position}
    output.append(position)
    
    # Start from 5
    if i < next_i:
        continue
        
    # Check if parabola can be built
    parabola_points = np.array([all_points[i-5], all_points[i-3], all_points[i-1]])
    if not isParabolaBuild(parabola_points):
        continue
    
    # If None for checked point, find next not None and use it
    for j in range(i, len(all_points) - 2):
        if (all_points[j] is not None) and (all_points[j+1] is not None) and (all_points[j+2] is not None):
            break
    if (all_points[j] is None) or (all_points[j+1] is None) or (all_points[j+2] is None):
        continue
            
    frame_number = j
           
    # Check trajectory if changed
    changed, res = checkTrajectory(parabola_points, all_points[j], all_points[j+1], all_points[j+2])

    '''
    if res[1] < res[0]:
        next_i = j + 6
    if res[2] < res[1]:
        next_i = j + 7
    if (res[1] < res[0]) or (res[2] < res[1]):
        continue
    '''
    
    if changed:
        # Check if the position was None
        '''
        if (j-1) in none_points:
            if res[0] < 3:
                continue
        '''
            
        # Check next frame (if not changed -> skip)
        '''
        if res[0] < 1:
            next_parabola_points = np.array([all_points[i-5], all_points[i-3], all_points[i-1]])
            if not isParabolaBuild(next_parabola_points):
                continue
                
            for k in range(j+1, len(all_points) - 2):
                if (all_points[k] is not None) and (all_points[k+1] is not None) and (all_points[k+2] is not None):
                    break
            
            if (all_points[k] is None) or (all_points[k+1] is None) or (all_points[k+2] is None):
                continue
                
            next_changed, _ = checkTrajectory(next_parabola_points, all_points[k], all_points[k+1], all_points[k+2]) 
            if not next_changed:
                continue
        '''

        # Calculate real trajectory change position
        new_i = j
        while (all_points[new_i-3] is None) or (all_points[new_i-1] is None):
            new_i -= 1
            if new_i == 0:
                continue
        if (all_points[new_i-3] is None) or (all_points[new_i-1] is None):
            continue
        
        line1 = np.array([all_points[i-3], all_points[i-1]])
        line2 = np.array([all_points[j], all_points[j+2]])
        
        if angle(line1[1] - line1[0], line2[1] - line2[0]) < 35:
            continue
        
        traj_change_position = calculateLineIntersection(line1, line2)
        
        # Calculate frame at which trajectory changed
        new_frame_number = getTrajectoryChangeFrame(frame_number, j, traj_change_position, all_points) 
        
        # Write trajectory change position
        traj_change_position = [round(traj_change_position[0], 1), round(traj_change_position[1], 1)]
        
        print(i, res)
        res_count += 1
        
        # Skip next 5 frames if trajectory change happened
        next_i = i + 5
        
    
    if traj_change_position is not None:
        key_frames.append({'frame': new_frame_number, 'key_position': traj_change_position})

output.append({'frame': len(all_points)-2, 'ball_position': all_points[len(all_points)-2], 'key_position': None})
output.append({'frame': len(all_points)-1, 'ball_position': all_points[len(all_points)-1], 'key_position': None})
    
for key_frame in key_frames:
    output[key_frame['frame']]['key_position'] = np.array(key_frame['key_position'])
    
# Set first not None as key frame
'''
for out in output:
    if out['ball_position'] is not None:
        out['key_position'] = out['ball_position']
        break
'''
print(res_count)

# Save data to file
saveCSVFile(output_file_path, output)