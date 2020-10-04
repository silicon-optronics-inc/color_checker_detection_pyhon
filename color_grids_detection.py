from cv2 import cv2 as cv
import numpy as np
from dataclasses import dataclass
import math
import os

TOTAL_COLUMNS = 6
TOTAL_ROWS = 4
TOTAL_GRIDS = TOTAL_COLUMNS * TOTAL_ROWS
SMALL_GRID_RATE_MAX = 0.9
SMALL_GRID_RATE_MIN = 0.35
SQUARE_CORRELATION_ENDURANCE = 0.2
PI = 3.14159
MIN_COLOR_GRID_NUM = 8
ANGLE_SUBTRACT_MAX_ENDURANCE = 1.6

CURRENT_DIR = os.getcwd()
output_path = CURRENT_DIR + '\\image\\'

# as a structure like C++
@dataclass
class SmallGridSizeRange:
    max_size: int
    min_size: int

def GetSmallGridSizeRange(Input_img):
    rows, cols, channel = Input_img.shape
    base_small_grid_size = cols * rows / TOTAL_GRIDS
    temp = SmallGridSizeRange(0,0)
    temp.max_size = base_small_grid_size * SMALL_GRID_RATE_MAX
    temp.min_size = base_small_grid_size * SMALL_GRID_RATE_MIN
    return temp

def FilterColorGrid(Input_contours, Grid_size_range):
    area = 0
    match_val = 0
    output_contours = []
    for contour in Input_contours:
        area = cv.contourArea(contour)
        match_val = cv.matchShapes(contour,np.array([[0,0],[1,0],[1,1],[0,1]]),cv.CONTOURS_MATCH_I2,0.0)
        if area < Grid_size_range.max_size and area > Grid_size_range.min_size and match_val < 1:
            output_contours.append(contour)
    return output_contours

def FindContourCenter(Input_contours):
    output_centroid = []
    for contour in Input_contours:
        my_moment = cv.moments(contour, False)
        centroid = np.array([my_moment['m10'] / my_moment['m00'], my_moment['m01'] / my_moment['m00']])
        output_centroid.append(centroid)
    return output_centroid

def FilterOutOverlapPoint(Input_contours, Center_positions):
    inside_contour_count = 0
    ret = 0
    output_centroid = []
    i=0
    for contour in Input_contours:
        inside_contour_count = 0
        i=0
        # for center_position in Center_positions:
        while i < len(Center_positions):
            ret = cv.pointPolygonTest(contour,(Center_positions[i][0],Center_positions[i][1]),False)
            if ret == 0 or ret == 1:
                if inside_contour_count >= 1:
                    None
                else:
                    output_centroid.append(Center_positions[i])
                inside_contour_count+=1
            i+=1
    return output_centroid

def GetRotationPoint(Org_point, Img_width, Img_height, Angle):
    org_point_new_coor = [0.0,0.0]
    rotation_point_new_coor = [0.0,0.0]
    rotation_point = [0.0,0.0]
    half_width = Img_width / 2
    half_height = Img_height / 2
    rad = (Angle * PI) / 180
    org_point_new_coor[0] = Org_point[0] - half_width
    if Org_point[1] <= half_height:
        org_point_new_coor[1] = half_height - Org_point[1]
    else:
        org_point_new_coor[1] = -1 * (Org_point[1] - half_height)

    #rotation matrix
    rotation_point_new_coor[0] = org_point_new_coor[0] * math.cos(rad) - org_point_new_coor[1] * math.sin(rad)
    rotation_point_new_coor[1] = org_point_new_coor[0] * math.sin(rad) + org_point_new_coor[1] * math.cos(rad)
    
    rotation_point[0] = rotation_point_new_coor[0] + half_width
    if rotation_point_new_coor[1] >= 0:
        rotation_point[1] = half_height - rotation_point_new_coor[1]
    else:
        rotation_point[1] = -1 * rotation_point_new_coor[1] + half_height

    return rotation_point

def DetectColorGrids(Color_checker_image):
    img_gray = cv.cvtColor(Color_checker_image,cv.COLOR_BGR2GRAY)

    img_denoise = cv.fastNlMeansDenoising(img_gray,10,7,21)

    img_threshold = cv.adaptiveThreshold(img_denoise,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,21,3)

    kernel = np.ones((3,3), np.uint8)
    img_eroded = cv.erode(img_threshold,kernel)

    small_grid_size_range = GetSmallGridSizeRange(Color_checker_image)

    contours, _hierarchy = cv.findContours(img_eroded, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    color_grid_contours = FilterColorGrid(contours, small_grid_size_range)

    centroid_positions = []
    centroid_positions = FindContourCenter(color_grid_contours)

    centroids_no_overlap = []
    centroids_no_overlap = FilterOutOverlapPoint(color_grid_contours,centroid_positions)

    #small to big, base on y axis
    centroids_no_overlap = sorted(centroids_no_overlap,key=lambda centroid: centroid[1])

    rad_top = math.atan((centroids_no_overlap[1][1] - centroids_no_overlap[0][1]) / 
                        (centroids_no_overlap[1][0] - centroids_no_overlap[0][0]))
    angle_top = rad_top * 180 / PI

    centroids_no_overlap_size = len(centroids_no_overlap)
    rad_bottom = math.atan((centroids_no_overlap[centroids_no_overlap_size - 1][1] - centroids_no_overlap[centroids_no_overlap_size - 2][1]) /
                            (centroids_no_overlap[centroids_no_overlap_size - 1][0] - centroids_no_overlap[centroids_no_overlap_size - 2][0]))
    angle_bottom = rad_bottom * 180 / PI

    angle_subtract = abs(angle_top - angle_bottom)

    if angle_subtract >= ANGLE_SUBTRACT_MAX_ENDURANCE:
        orientation_left_top = centroids_no_overlap[0]
        orientation_right_bottom=[0.0,0.0]

        for centroid in centroids_no_overlap:
            if centroid[0] < orientation_left_top[0]:
                orientation_left_top[0] = centroid[0]
            if centroid[1] < orientation_left_top[1]:
                orientation_left_top[1] = centroid[1]
            if centroid[0] > orientation_right_bottom[0]:
                orientation_right_bottom[0] = centroid[0]
            if centroid[1] > orientation_right_bottom[1]:
                orientation_right_bottom[1] = centroid[1]

        translation=[0.0,0.0]
        translation[0] = (orientation_right_bottom[0] - orientation_left_top[0]) / (TOTAL_COLUMNS - 1)
        translation[1] = (orientation_right_bottom[1] - orientation_left_top[1]) / (TOTAL_ROWS - 1)
        
        grids_position = np.zeros((TOTAL_ROWS, TOTAL_COLUMNS, 2), dtype=np.int)

        row_count = 0
        col_count = 0
        while(row_count < TOTAL_ROWS):
            col_count = 0
            while(col_count < TOTAL_COLUMNS):
                grids_position[row_count][col_count][0] = orientation_left_top[0] + translation[0]*col_count
                grids_position[row_count][col_count][1] = orientation_left_top[1] + translation[1]*row_count
                col_count += 1
            row_count += 1
        
        return grids_position

    else:
        angle_avg = (angle_top + angle_bottom) / 2
        img_rotation = np.zeros(img_gray.shape, dtype=np.uint8)
        rows, cols = img_gray.shape
        center = [cols / 2, rows / 2]
        rotation_mat = cv.getRotationMatrix2D((center[0],center[1]),angle_avg,1.0)
        img_rotation = cv.warpAffine(img_eroded,rotation_mat,(cols,rows))

        contours, _hierarchy = cv.findContours(img_rotation, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        color_grid_contours = FilterColorGrid(contours, small_grid_size_range)

        centroid_positions = []
        centroid_positions = FindContourCenter(color_grid_contours)

        orientation_left_top = centroid_positions[0]
        orientation_right_bottom=[0.0,0.0]

        # contour_arr_all = np.zeros(img_gray.shape, dtype=np.uint8)

        i=0
        for centroid in centroid_positions:
            if centroid[0] < orientation_left_top[0]:
                orientation_left_top[0] = centroid[0]
            if centroid[1] < orientation_left_top[1]:
                orientation_left_top[1] = centroid[1]
            if centroid[0] > orientation_right_bottom[0]:
                orientation_right_bottom[0] = centroid[0]
            if centroid[1] > orientation_right_bottom[1]:
                orientation_right_bottom[1] = centroid[1]
            i+=1

        translation=[0.0,0.0]
        translation[0] = (orientation_right_bottom[0] - orientation_left_top[0]) / (TOTAL_COLUMNS - 1)
        translation[1] = (orientation_right_bottom[1] - orientation_left_top[1]) / (TOTAL_ROWS - 1)

        grid_coordinate = np.zeros((TOTAL_ROWS, TOTAL_COLUMNS, 2), dtype=np.int)

        row_count = 0
        col_count = 0
        while(row_count < TOTAL_ROWS):
            col_count = 0
            while(col_count < TOTAL_COLUMNS):
                grid_coordinate[row_count][col_count][0] = orientation_left_top[0] + translation[0]*col_count
                grid_coordinate[row_count][col_count][1] = orientation_left_top[1] + translation[1]*row_count
                col_count += 1
            row_count +=1

        grids_position = np.zeros((TOTAL_ROWS, TOTAL_COLUMNS, 2), dtype=np.int)
        temp_coordinate = [0,0]
        row_count = 0
        col_count = 0
        while(row_count < TOTAL_ROWS):
            col_count = 0
            while(col_count < TOTAL_COLUMNS):
                temp_coordinate = GetRotationPoint((grid_coordinate[row_count][col_count][0],grid_coordinate[row_count][col_count][1]), cols, rows, -1 * angle_avg)
                grids_position[row_count][col_count][0] = temp_coordinate[0]
                grids_position[row_count][col_count][1] = temp_coordinate[1]
                col_count += 1
            row_count += 1

        return grids_position
