import numpy
import cv2
import os

from picture_sudoku.helpers import cv2_helper
from picture_sudoku.helpers import numpy_helper

# large is brighter, less is darker.
BLACK = cv2_helper.BLACK
WHITE = cv2_helper.WHITE

IMG_SIZE = 32
SUDOKU_COUNT = 9

def find_max_contour(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.0001):
    contours = cv2_helper.find_contours(
        threshed_image, filter_func, accuracy_percent_with_perimeter)
    if len(contours) == 0:
        return None
    contour_area_arr = [cv2.contourArea(i) for i in contours]
    max_contour = contours[contour_area_arr.index(max(contour_area_arr))]
    return max_contour

def find_sudoku_sqare(threshed_image):
    ''' 
        it's really hard.
        It's very depend on the threshed_image.
    '''
    max_contour = find_max_contour(threshed_image)
    for i in reversed(range(1,10)):
        result = get_approximated_contour(max_contour, 0.01*i)
        # i.p()
        # len(result).p()
        if len(result)==4:
            return result
    return None

def find_reasonable_lines(the_image, line_count_threshold=50):
    def find(low_threshold):
        cur_image = the_image.copy()
        high_threshold = low_threshold * 2
        # low_threshold,high_threshold = 50, 100

        cur_image.shape.pp()
        threshold = int(cur_image.shape[0] * 0.4)
        threshold.pp()

        cur_image = cv2.Canny(cur_image, low_threshold, high_threshold)

        lines = cv2.HoughLines(cur_image, rho=1, theta=numpy.pi/180, threshold= threshold)
        ''' this is true line list'''
        if numpy_helper.is_array_none(lines):
            return []
        return lines[0]


    lines = []
    for low_threshold in range(130,200,10):
        lines = find(low_threshold)
        if len(lines) < 30:
            break
    len(lines).pp()
    low_threshold.pp()
    show_lines(the_image, lines)
    return lines


def show_lines(the_image, lines):
    color_image = cv2.cvtColor(the_image.copy(), cv2.COLOR_GRAY2BGR)
    for rho, theta in lines:
        # (rho, theta).pp()
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho
        point0 = (int(numpy.around(x0 + 1000*(- sin_theta))),  int(numpy.around(y0 + 1000*( cos_theta))))
        point1 = (int(numpy.around(x0 - 1000*(- sin_theta))),  int(numpy.around(y0 - 1000*( cos_theta))))
        cv2.line(color_image, point0, point1, (0,0,255), thickness=2)
    cv2_helper.Image.show(color_image)
    # return color_image

def find_lines_by_standard_hough(the_image):
    cur_image = the_image.copy()
    ''' This blur operation is quite useful '''
    # cur_image = cv2.GaussianBlur(cur_image,ksize=(5,5), sigmaX=0)
    # cur_image = cv2.GaussianBlur(cur_image,ksize=(3,3), sigmaX=0)

    ''' open, it could remove the border '''
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cur_image = cv2.morphologyEx(cur_image, cv2.MORPH_OPEN, kernel)


    # cv2_helper.Image.show(cur_image)
    # low_threshold, high_threshold = 40, 80
    low_threshold, high_threshold = 70, 140

    cur_image = cv2.Canny(cur_image, low_threshold, high_threshold)

    # cv2_helper.Image.show(cur_image)
    lines = cv2.HoughLines(cur_image, rho=1, theta=numpy.pi/180, threshold= 200)
    ''' this is true line list'''
    if numpy_helper.is_array_none(lines):
        return []
    lines = lines[0]
    len(lines).pp()
    ''' you can decline the count line by canny low_threshold '''
    if len(lines) > 200 or len(lines) < 1:
        return []

    show_lines(the_image, lines)


def find_lines_by_probabilistic_hough(the_image):
    cur_image = the_image.copy()

    ''' This blur operation is quite useful '''
    # cur_image = cv2.GaussianBlur(cur_image,ksize=(5,5), sigmaX=0)
    cur_image = cv2.GaussianBlur(cur_image,ksize=(3,3), sigmaX=0)

    ''' open, it could remove the border '''
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cur_image = cv2.morphologyEx(cur_image, cv2.MORPH_OPEN, kernel)


    low_threshold, high_threshold = 50, 140

    cur_image = cv2.Canny(cur_image, low_threshold, high_threshold)
    cv2_helper.Image.show(cur_image)

    lines = cv2.HoughLinesP(cur_image, rho=1, theta=numpy.pi/180, 
        threshold= 100, minLineLength=30, maxLineGap=10)
    lines = lines[0]
    len(lines).pp()
    ''' you can decline the count line by canny low_threshold '''
    if len(lines) > 100:
        return color_image

    for line in lines:
        # line.pp()
        point0 = (line[0], line[1])
        point1 = (line[2], line[3])
        cv2.line(color_image, point0, point1, (0,0,255), thickness=2)
    cv2_helper.Image.show(color_image)


def find_sudoku_number_binary_arr(gray_image):
    the_image = gray_image

    threshed_pic_array = cv2.adaptiveThreshold(the_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    # cv2_helper.Image.show(the_image)

    square_contour = find_max_contour(threshed_pic_array)

    # cv2_helper.Image.show_contours_with_color(the_image, [square_contour])

    square_ragion = cv2_helper.get_rect_ragion_with_contour(gray_image, square_contour)

    # find_lines_by_standard_hough(square_ragion)
    # find_lines_by_probabilistic_hough(square_ragion)
    find_reasonable_lines(square_ragion)


if __name__ == '__main__':
    from minitest import *

    def show_square(image_path=None):
        image_path.pp()
        gray_image = cv2.imread(image_path, 0)
        color_image  = cv2.imread(image_path)
        # gray_image = cv2_helper.resize_with_fixed_height(gray_image)
        # color_image = cv2_helper.resize_with_fixed_height(color_image)
        find_sudoku_number_binary_arr(gray_image)

    with test("show_square"):
        # show_square('../resource/example_pics/sample04.dataset.jpg')
        # for i in range(1,15):
        for i in range(1,5):
            image_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
            show_square(image_path)

        pass