import numpy as np
from cv2 import cv2 as cv
from color_grids_detection import DetectColorGrids
import os

TOTAL_COLUMNS = 6
TOTAL_ROWS = 4

CURRENT_DIR = os.getcwd() 

img=cv.imread(CURRENT_DIR + '\\image\\image.jpg')
output_path = CURRENT_DIR + '\\image\\'
cascade_path=CURRENT_DIR + '\\cascade.xml'

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray=cv.equalizeHist(gray)
cascade=cv.CascadeClassifier(cascade_path)
objects=cascade.detectMultiScale(gray,1.1,6,1,(129,90))
for single_obj in objects:
    cv.rectangle(img,single_obj,(255,255,255))
    x, y, w, h = single_obj
    crop_img = img[y:y+h,x:x+w]

    grids_position = np.zeros((TOTAL_ROWS, TOTAL_COLUMNS, 2), dtype=np.int)

    grids_position = DetectColorGrids(crop_img)

    row_count = 0
    col_count = 0
    while(row_count < TOTAL_ROWS):
        col_count = 0
        while(col_count < TOTAL_COLUMNS):
            cv.circle(img,(grids_position[row_count][col_count][0] + x, grids_position[row_count][col_count][1] + y),7,[255,255,255],2)
            col_count += 1
        row_count += 1

    cv.imwrite(output_path+'image-grid.jpg', img)
    print('Finished!')