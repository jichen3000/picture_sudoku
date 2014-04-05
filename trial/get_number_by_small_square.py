import numpy
import cv2
import os

from picture_sudoku.helpers import cv2_helper
from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

# large is brighter, less is darker.
BLACK = cv2_helper.BLACK
WHITE = cv2_helper.WHITE

IMG_SIZE = 32
SUDOKU_SIZE = 9
SMALL_COUNT = 3

def find_max_contour(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.0001):
    contours = cv2_helper.Image.find_contours(
        threshed_image, filter_func, accuracy_percent_with_perimeter)
    if len(contours) == 0:
        return None
    contour_area_arr = [cv2.contourArea(i) for i in contours]
    max_contour = contours[contour_area_arr.index(max(contour_area_arr))]
    return max_contour

def get_approximated_contour(contour, accuracy_percent_with_perimeter=0.0001):
    perimeter = cv2.arcLength(contour,True)
    # notice the approximation accuracy is the key, if it is 0.01, you will find 4
    # if it is larger than 0.8, you will get nothing at all.
    approximation_accuracy = accuracy_percent_with_perimeter*perimeter
    return cv2.approxPolyDP(contour,approximation_accuracy,True)


def convert_to_square(contour):
    for i in reversed(range(1,10)):
        result = get_approximated_contour(contour, 0.01*i)
        # i.p()
        # len(result).p()
        if len(result)==4:
            return result
    return None



def warp_square(pic_array, approximated_square):
    ''' 
        let the approximated_square become a true square.
        
    '''
    square_rect = cv2.boundingRect(approximated_square)
    square_rect = cv2_helper.Rect.adjust_to_minimum(square_rect)
    # square_rect_contour = cv2_helper.rect_to_contour(square_rect)

    # approximated_square.pp()
    # square_rect_contour.pp()

    # approximated_square.shape is (4,1,2)
    approximated_square_float = numpy.float32(approximated_square.copy())
    approximated_square_float = cv2_helper.Quadrilateral.vertices(approximated_square_float)
    # square_rect_contour_float = numpy.float32(square_rect_contour.copy())
    square_rect_contour_float = numpy.float32(cv2_helper.Rect.vertices(square_rect))

    retval = cv2.getPerspectiveTransform(approximated_square_float,square_rect_contour_float)
    dsize = pic_array.shape[::-1]
    warp = cv2.warpPerspective(pic_array,retval,dsize)
    return warp, square_rect



def find_small_square_contour(the_image, outer_square_contour, square_count_in_row, is_blur=False):
    '''
        Now, square_count_in_row will be SUDOKU_SIZE AKA 9 or SUDOKU_SIZE 3
    '''
    square_count = square_count_in_row ** 2
    if is_blur:
        # the_image = cv2.GaussianBlur(the_image,ksize=(5,5), sigmaX=0)
        the_image = cv2.bilateralFilter(the_image, 5, 10, 3)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # the_image = cv2.morphologyEx(the_image, cv2.MORPH_CLOSE, kernel)
    # the_image = cv2.morphologyEx(the_image, cv2.MORPH_OPEN, kernel)
    # the_image = cv2.erode(the_image, kernel, iterations=2)
    # the_image = cv2.dilate(the_image, kernel, iterations=1)
    # the_image = cv2.adaptiveThreshold(the_image,WHITE,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)


    low_threshold, high_threshold = 40, 80
    # low_threshold, high_threshold = 80, 250

    the_image = cv2.Canny(the_image, low_threshold, high_threshold)
    # cv2_helper.Image.show(the_image)


    # the_image = cv2.adaptiveThreshold(the_image,WHITE,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)
    # cv2_helper.Image.show(the_image)




    expected_perimeter = cv2.arcLength(outer_square_contour,True) / square_count_in_row
    expected_area = cv2.contourArea(outer_square_contour) / square_count
    def filter_perimeter(contour):
        hull = cv2.convexHull(contour)
        # hull = contour
        # return True
        # return cv2.arcLength(hull,True) > expected_perimeter * 0.9
        perimeter_flag = list_helper.is_in_range(cv2.arcLength(hull,True), expected_perimeter, 0.3)
        if not perimeter_flag:
            return False
        area = cv2.contourArea(hull)
        return list_helper.is_in_range(area, expected_area, 0.3)



    contours = cv2_helper.Image.find_contours(
        the_image, filter_perimeter, 0.0001)
    # len(contours).pp()
    contours = map(cv2.convexHull, contours)
    # contours = map(convert_to_square, contours)
    # cv2_helper.Image.show_contours_with_color(the_image, contours)

    for contour in contours:
        cv2_helper.Image.show_contours_with_color(the_image, [contour])
    return contours

def find_small_square_contour_original(the_image, outer_square_contour, square_count_in_row, is_blur=False):
    '''
        Now, square_count_in_row will be SUDOKU_SIZE AKA 9 or SUDOKU_SIZE 3
    '''
    square_count = square_count_in_row ** 2
    if is_blur:
        the_image = cv2.GaussianBlur(the_image,ksize=(5,5), sigmaX=0)

    the_image = cv2.adaptiveThreshold(the_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)
    cv2_helper.Image.show(the_image)

    expected_perimeter = cv2.arcLength(outer_square_contour,True) / square_count_in_row
    expected_area = cv2.contourArea(outer_square_contour) / square_count
    def filter_perimeter(contour):
        # return True
        perimeter_flag = list_helper.is_in_range(cv2.arcLength(contour,True), expected_perimeter, 0.3)
        if not perimeter_flag:
            return False
        area = cv2.contourArea(contour)
        return list_helper.is_in_range(area, expected_area, 0.3)



    contours = cv2_helper.Image.find_contours(
        the_image, filter_perimeter, 0.001)

    # len(contours).pp()
    cv2_helper.Image.show_contours_with_color(the_image, contours)
    # for contour in contours:
    #     cv2_helper.Image.show_contours_with_color(the_image, [contour])
    return contours

def speculate_lost_squares():
    ''' reshape to matrix'''


def sort_squares_as_sudoku(squares, max_square):
    ''' '''
    square_centers = map(cv2_helper.Quadrilateral.center, squares)
    # [s[0] for s in square_centers].pp()
    # max_square.pp()
    top_left, bottom_left, bottom_right, top_right = \
            cv2_helper.Quadrilateral.vertices(max_square)
    y_delta = abs((bottom_right[1] - bottom_left[1]) - (top_right[1] - top_left[1]))
    x_delta = abs((top_right[0] - bottom_right[0]) - (top_left[0] - bottom_left[0]))

    x_cat = []
    y_cat = []
    for center_x, center_y in square_centers:
        x_key_not_exist = True
        for key, values in x_cat:
            if abs(key - center_x) < x_delta:
                values.append((center_x, center_y))
                x_key_not_exist = False
        if x_key_not_exist:
            x_cat.append([center_x, [(center_x, center_y)]])



def find_sudoku_number_binary_arr(gray_image):
    the_image = gray_image

    threshed_image = cv2.adaptiveThreshold(the_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    # cv2_helper.Image.show(threshed_image)

    ''' 
        it's really hard.
        It's very depend on the threshed_image.
    '''
    max_contour = find_max_contour(threshed_image)
    square_contour = convert_to_square(max_contour)

    # cv2_helper.Image.show_contours_with_color(the_image, [square_contour])

    larged_contour = cv2_helper.Quadrilateral.enlarge(square_contour, 0.02)
    square_ragion = cv2_helper.Contour.get_rect_ragion(larged_contour, the_image)

    # cv2_helper.Image.show(square_ragion)


    small_contours = find_small_square_contour(square_ragion, square_contour, SMALL_COUNT, is_blur=True)
    if len(small_contours) < SMALL_COUNT**2 / 2.0:
        len(small_contours).p("Analyzing large squares is failed.")
        small_contours = find_small_square_contour(square_ragion, square_contour, SUDOKU_SIZE)
        if len(small_contours) < SUDOKU_SIZE**2 / 2.0:
            len(small_contours).p("Analyzing small squares is also failed.")
            return False
    len(small_contours).p()

    ''' to square which only have tow vertices'''
    # small_squares = map(convert_to_square, small_contours)
    # small_squares.size().pp()
    # cv2_helper.Image.show_contours_with_color(square_ragion, small_squares)

    ''' sort them as sudoku '''
    # sort_squares_as_sudoku(small_squares,square_contour)
    ''' union the squares which in the same ragion'''
    ''' speculate the lost squares'''





if __name__ == '__main__':
    from minitest import *

    def show_square(image_path=None):
        image_path.pp()
        gray_image = cv2.imread(image_path, 0)
        color_image  = cv2.imread(image_path)
        gray_image = cv2_helper.Image.resize_keeping_ratio_by_height(gray_image)
        color_image = cv2_helper.Image.resize_keeping_ratio_by_height(color_image)
        find_sudoku_number_binary_arr(gray_image)

    with test("show_square"):
        # 5, iterations = 2
        # 6, 1
        # 7, 1, but jsut get 4
        show_square('../resource/example_pics/sample07.dataset.jpg')
        # for i in range(1,15):
        #     image_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        #     show_square(image_path)

        pass