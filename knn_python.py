""" K-NEAREST NEIGHBORS SIMULATION IN PYTHON """
""" WRITTEN BY DOGAN YIGIT YENIGUN (toUpperCae78) """
""" VERSION 1.0 """
from scipy.spatial import ConvexHull
from datetime import datetime
import numpy as np
import random
import math
import cv2

# The Euclidean distance method is responsible for performing distance calculations
def euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Create data objects up to the specified amount and by randomly assigning (x,y) values on the surface
def create_data_objects():
    dobj = []
    for i in range(100):
        dobj.append([i, (random.randint(0,width-1), random.randint(0,height-1))])
    return dobj

# Create the query object by randomly assigning (x, y) value on the surface
def create_query_object():
    qx = random.randint(0,width-1);   qy = random.randint(0,height-1)
    return qx, qy

# Perform distance calculations for the query object
# Some lists must be initialized when new set of data objects are generated from scratch or the query object was moved
def calculate_knn_query():
    dist_q = []             # The distances of all data objects against the query objects
    shortest_dist = []      # The shortest k distances to the query objects
    shortest_dist_no = []   # The data objects numbers that are the k neartest to the query object
    is_nearest = []         # Determine whether each data object is one of the k nearest

    # Put initial values to these distance-related arrays (assigning max distances & non-existent data object numbers)
    for i in range(k):
        shortest_dist.append(1000000 - k + i + 1)
        shortest_dist_no.append(data_cnt + i)

    # Initialize the status of being nearest for all data objects to query as False
    # Then, get the distances againsts the query object
    for i in range(data_cnt):
        is_nearest.append(False)
        dist_q.append(euclidean_distance(dobj[i][1][0], qx, dobj[i][1][1], qy))
        # If the distance for the data object is one of the k shortest so far, add it to the shortest distance list
        # plus its number to the ohter list, and remove the rightmost one
        for j in range(k):
            if dist_q[i] < shortest_dist[j]:
                shortest_dist.insert(j, dist_q[i])
                shortest_dist.pop()
                shortest_dist_no.insert(j, i)
                shortest_dist_no.pop()
                break
    
    # Mark all k nearest objects as True to be shown properly on the surface
    for i in range(k):
        is_nearest[shortest_dist_no[i]] = True

    # Get the maximum distance among all k nearest data objects to determine the radius of query range
    shortest_dist_max = max(shortest_dist)

    # If show_distance = True, output the k shortest distances in the output
    # FORMAT: [(<Data object no>, <Distance to the query object>), ...]
    if show_distance:
        print("Shortest Distances = {}".format(list(zip(shortest_dist_no, shortest_dist))))

    return shortest_dist_no, shortest_dist_max

# Perform distances calculations for all data objects
# Some lists must be initialized every time the method is called
# Results are always shown in the output when new new set of data objects are generated or k was changed
def calculate_knn_data_objects():
    dobj_id = []     # Consecutive data object numbers are stored here
    dobj_knn = []    # The shortest k distances for each data objects against the others

    for i in range(data_cnt):
        dist_dobj = []               # The distance of all data objects against one data object
        shortest_dist_dobj = []      # The shortest k distances to one data object
        shortest_dist_no_dobj = []   # The data object numbers that are the k nearest to one data object

        # Put initial values to these two arrays (assigning max distances & non-existent data objects)
        for j in range(k):
            shortest_dist_dobj.append(1000000 - k + j + 1)
            shortest_dist_no_dobj.append(data_cnt + j)
        
        # The distance calculations are roughly the same as those for the query object
        for j in range(data_cnt):
            if j == i:
                dist_dobj.append(1000000)
            else:
                dist_dobj.append(euclidean_distance(dobj[j][1][0], dobj[i][1][0], dobj[j][1][1], dobj[i][1][1]))
                for m in range(k):
                    if dist_dobj[j] < shortest_dist_dobj[m]:
                        shortest_dist_dobj.insert(m, dist_dobj[j])
                        shortest_dist_dobj.pop()
                        shortest_dist_no_dobj.insert(m, j)
                        shortest_dist_no_dobj.pop()
                        break

        # After the kNN result is ready for the related data object, add them to more generic arrays
        get_result = []
        for j in range(len(shortest_dist_no_dobj)):
            get_result.append(shortest_dist_no_dobj[j])
        dobj_id.append(i)
        dobj_knn.append(get_result)

    # Show the output for each data object
    print("Data Objects kNN = {}".format(list(zip(dobj_id, dobj_knn))))
    return dobj_id, dobj_knn

# Get the convex hull shape that is derived from data objects
# Achieved by SciPy's ConvexHull method
def convex_hull():
    points = []
    for i in range(data_cnt):
        points.append([dobj[i][1][0], dobj[i][1][1]])
    points = np.array(points)
    hull = ConvexHull(points)

    print(f"Convex Hull Points = {hull.simplices.tolist()}")
    return hull.simplices

# Left or right mouse clicks on the surface will reveal the k nearest data objects and distances at the specified coordinate
def mouse_click(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        if my < height:
            dist = []
            for i in range(data_cnt):
                dist.append((i, euclidean_distance(mx, dobj[i][1][0], my, dobj[i][1][1])))
            dist_sorted = sorted(dist, key=lambda x: x[1])
            print(f"x = {mx} | y = {my} --> {dist_sorted[:k]}")

# Initial values and methods to execute for the kNN simulation
print("### K-Nearest Neighbors in Python ###")
width = 1024;   height = 768
data_cnt = 40;   qspeed = 8;   k = 3
show_numbers = False;   show_coords = False;   show_distance = False;   show_connection = False
show_convex_hull = False;   dark_mode = False
print(f"Area Resolution = {width} x {height} | Data Objects = {data_cnt} | k = {k} | Query Speed = {qspeed} px")
dobj = create_data_objects()
qx, qy = create_query_object()
shrt_dist_no, shrt_dist_max = calculate_knn_query()
hull_points = convex_hull()
dobj_id, dobj_knn = calculate_knn_data_objects()
cv2.namedWindow("K-Nearest Neighbors in Python")

# All surface painting operations are carried out here and update necessarily when an action was performed
img = np.uint8(np.zeros((height+90, width, 3)))
while True:
    # Light gray surface is adopted in light mode
    if dark_mode:    img[:, :, :] = 24
    else:   img[:, :, :] = 224

    # Show all the data objects as small black circles and also show their numbers and coords when enabled
    # The color becomes blue when it is one of the k nearest objects, also a line is drawn accordingly
    for i in range(data_cnt):
        if i in shrt_dist_no:    
            obj_color = (255, 100, 0)
            cv2.line(img, (qx, qy), (dobj[i][1][0], dobj[i][1][1]), obj_color, 1, cv2.LINE_AA)
        else:    
            if dark_mode:   obj_color = (255, 255, 255)
            else:   obj_color = (0, 0, 0)
        cv2.circle(img, (dobj[i][1][0],dobj[i][1][1]), 4, obj_color, -1, cv2.LINE_AA)
        if show_numbers:
            if i < 10:   dnsx = -18
            else:   dnsx = -29
            cv2.putText(img, str(i), (dobj[i][1][0]+dnsx, dobj[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj_color, 1, cv2.LINE_AA)
        if show_coords:
            if dobj[i][1][0] > width - 100 and dobj[i][1][1] > height - 25:   dcsx = -75;   dcsy = -10
            elif dobj[i][1][0] > width - 100:   dcsx = -75;   dcsy = 19
            elif dobj[i][1][1] > height - 25:   dcsx = 5;   dcsy = -10
            else:    dcsx = 5;   dcsy = 19
            cv2.putText(img, "("+str(dobj[i][1][0])+","+str(dobj[i][1][1])+")", (dobj[i][1][0]+dcsx, dobj[i][1][1]+dcsy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 1, cv2.LINE_AA)
    
    # Show the query object as slightly bigger red circle and show coords when enabled
    cv2.circle(img, (qx, qy), 6, (0, 0, 255), -1, cv2.LINE_AA)
    if show_coords:
        if qx > width - 100 and qy > height - 25:    qsx = -75;   qsy = -10
        elif qx > width - 100:    qsx = -75;   qsy = 19
        elif qy > height - 25:    qsx = 5;     qsy = -10
        else:   qsx = 5;   qsy = 19
        cv2.putText(img, "("+str(qx)+","+str(qy)+")", (qx+qsx, qy+qsy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw the query range circle in red color
    # The radius is equal to the maximum shortest distance among all k nearest data objects
    cv2.circle(img, (qx, qy), int(shrt_dist_max), (0, 0, 255), 1, cv2.LINE_AA)

    # If enabled, show kNN connections for all data objects to each other in green color
    if show_connection:
        for i in range(len(dobj_knn)):
            result = dobj_knn[i]
            for j in range(len(result)):
                cv2.line(img, (dobj[i][1][0], dobj[i][1][1]), (dobj[result[j]][1][0], dobj[result[j]][1][1]), (0, 180, 0), 1, cv2.LINE_AA)

    # If enabled, show the convex hull derived from the data objects in purple color
    if show_convex_hull:
        for i in range(len(hull_points)):
            l = hull_points[i][0];   r = hull_points[i][1]
            cv2.line(img, (dobj[l][1][0], dobj[l][1][1]), (dobj[r][1][0], dobj[r][1][1]), (100, 0, 100), 2, cv2.LINE_AA)

    if dark_mode:   
        img[height:, :, :] = 36;    text_color = (255, 255, 255)
    else:   
        img[height:, :, :] = 240;   text_color = (0, 0, 0)
    cv2.putText(img, "k="+str(k)+" | Query Sp="+str(qspeed), (10, height+25), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Data Objects="+str(data_cnt), (10, height+50), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(img, "SPACE = Create New | Arrow Keys = Move Query | Z / X = Change k | C / V = Change Query Sp", 
                (200, height+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(img, "B / N = Change Data Obj Number | A = Show Numbers | S = Show Coords | D = Show Distance",
                (200, height+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(img, "F = Show Connections | G = Show Convex Hull | Q = Light / Dark Mode",
                (200, height+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.imshow("K-Nearest Neighbors in Python", img)
    cv2.setMouseCallback("K-Nearest Neighbors in Python", mouse_click)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:     # ESC - Quit the simulation
        break
    elif key == 32:   # Space - Create new data objects
        print("Created new set of data objects")
        dobj = create_data_objects()
        qx, qy = create_query_object()
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
        hull_points = convex_hull()
        dobj_id, dobj_knn = calculate_knn_data_objects()
    elif key == ord('2'):       # Number 2 - Query object moving down
        qy += qspeed
        if qy > height:   qy = height
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
    elif key == ord('4'):       # Number 4 - Query object moving left
        qx -= qspeed
        if qx < 0:   qx = 0
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
    elif key == ord('6'):       # Number 6 - Query object moving right
        qx += qspeed
        if qx > width:   qx = width
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
    elif key == ord('8'):       # Number 8 - Query object moving up
        qy -= qspeed
        if qy < 0:   qy = 0
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
    elif key == ord('A') or key == ord('a'):    # A - Show/hide numbers of data objects
        show_numbers = not show_numbers
        if show_numbers:   print("Enabled showing data object numbers")
        else:   print("Disabled showing data object numbers")
    elif key == ord('S') or key == ord('s'):    # S - Show/hide coordinates of data objects & query object
        show_coords = not show_coords
        if show_coords:   print("Enabled showing data object coordinates")
        else:   print("Disabled showing data object coordinates")
    elif key == ord('D') or key == ord('d'):    # D - Show/hide distances of k nearest data objects
        show_distance = not show_distance
        if show_distance:   print("Enabled k-nearest objects distance output")
        else:   print("Disabled k-nearest objects distance output")
    elif key == ord('F') or key == ord('f'):    # F - Show/hide connections for all data objects (based on their KNN)
        show_connection = not show_connection
        if show_connection:   print("Enabled data objects kNN connections")
        else:   print("Disabled data objects kNN connections")
    elif key == ord('G') or key == ord('g'):    # G - Show/hide the convex hull (derived from the data objects)
        show_convex_hull = not show_convex_hull
        if show_convex_hull:   print("Enabled showing convex hull")
        else:   print("Disabled showing convex hull")
    elif key == ord('Q') or key == ord('q'):    # Q - Toggle the light/dark mode
        dark_mode = not dark_mode
        if dark_mode:   print("Switched to dark mode")
        else:   print("Switched to light mode")
    elif key == ord('Z') or key == ord('z'):    # Z - Decrease k value by 1 (minimum = 1)
        if k > 1:   k -= 1
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
        dobj_id, dobj_knn = calculate_knn_data_objects()
        print(f"k = {k}")
    elif key == ord('X') or key == ord('x'):    # X - Increase k value by 1 (maximum = 10)
        if k < 10:   k += 1
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
        dobj_id, dobj_knn = calculate_knn_data_objects()
        print(f"k = {k}")
    elif key == ord('C') or key == ord('c'):    # C - Decrease the speed of query object by 1 (minimum = 1)
        if qspeed > 1:   qspeed -= 1
        print(f"Query speed = {qspeed} px")
    elif key == ord('V') or key == ord('v'):    # V - Increase the speed of query object by 1 (maximum = 20)
        if qspeed < 20:   qspeed += 1
        print(f"Query speed = {qspeed} px")
    elif key == ord('B') or key == ord('b'):    # B - Decrease the number of visible data objects by 5 (minimum = 10)
        if data_cnt > 10:   data_cnt -= 5
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
        hull_points = convex_hull()
        dobj_id, dobj_knn = calculate_knn_data_objects()
        print(f"Data Objects = {data_cnt}")
    elif key == ord('N') or key == ord('n'):    # N - Increase the number of visible data objects by 5 (maximum = 100)
        if data_cnt < 100:   data_cnt += 5
        shrt_dist_no, shrt_dist_max = calculate_knn_query()
        hull_points = convex_hull()
        dobj_id, dobj_knn = calculate_knn_data_objects()
        print(f"Data Objects = {data_cnt}")
    elif key == ord('P') or key == ord('p'):    # P - Save as image
        cr_date = datetime.now()
        filename = "knn_python_{}_{}_{}_{}_{}_{}.jpg".format(
            cr_date.year, cr_date.month, cr_date.day, cr_date.hour, cr_date.minute, cr_date.second)
        cv2.imwrite(filename, img)
        print("The kNN image was saved as", filename)

cv2.destroyAllWindows()