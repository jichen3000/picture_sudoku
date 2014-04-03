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

def filter_number_contour(the_contour, the_image):
    height, width = the_image.shape
    cell_height = height / 9 *0.85
    cell_width = width / 9 *0.7
    x,y, contour_width, contour_height = cv2.boundingRect(the_contour)
    # number_ragion_area = height * width / SUDOKU_COUNT ** 2
    # area_flag = cv2.contourArea(the_contour) < number_ragion_area
    contour_area = (contour_width * contour_height)
    cell_area =  (height * width / SUDOKU_COUNT ** 2)
    area_flag = cell_area > contour_area > cell_area * 0.04
    # area_flag = cv2.contourArea(the_contour) > 0.01 * cell_area
    # area_flag = cell_area > contour_area
    length_flag = (cell_height > contour_height and cell_width > contour_width)
    # length_flag = True
    ''' it would remove the border line'''
    ratio_flag = contour_width / contour_height < 4 and contour_height / contour_width < 8
    # ratio_flag = True
    # if cell_area > contour_area and length_flag:
    #     "new".p()
    #     area_flag.p()
    #     contour_area.p()
    #     cell_area.p()
    return length_flag and area_flag and ratio_flag

def find_number_contours(the_image):
    # cv2_helper.Image.show(the_image)
    ''' This blur operation is quite useful '''
    the_image = cv2.GaussianBlur(the_image,ksize=(5,5), sigmaX=0)

    ''' open, it could remove the border '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    the_image = cv2.morphologyEx(the_image, cv2.MORPH_OPEN, kernel)
    # the_image = cv2.erode(the_image, kernel)
    # cv2_helper.Image.show(the_image)
    # the_image = cv2.adaptiveThreshold(the_image,WHITE,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)


    low_threshold, high_threshold = 40, 80

    the_image = cv2.Canny(the_image, low_threshold, high_threshold)

    ''' change the image to uint8'''
    the_image = cv2.convertScaleAbs(the_image)

    contours = cv2_helper.find_contours(the_image, None, 0.01)
    cv2_helper.Image.show_contours_with_color(the_image, contours)
    contours = filter(lambda c: filter_number_contour(c, the_image), contours)
    return contours

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


def find_sudoku_number_binary_arr(gray_image):
    the_image = gray_image

    threshed_pic_array = cv2.adaptiveThreshold(the_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    # cv2_helper.Image.show(threshed_pic_array)

    square_contour = find_sudoku_sqare(threshed_pic_array)

    # cv2_helper.Image.show_contours_with_color(the_image, [square_contour])

    warp, square_rect = warp_square(the_image, square_contour)
    square_contour = cv2_helper.Rect.to_contour(square_rect)


    square_ragion = cv2_helper.Rect.get_ragion(square_rect, warp)
    # cv2_helper.Image.show(square_ragion)

    number_contours = find_number_contours(square_ragion)
    len(number_contours).pp()
    cv2_helper.Image.show_contours_with_color(square_ragion, number_contours)
    # for contour in number_contours:
    #     cv2.contourArea(contour).pp()
    #     cv2_helper.Image.show_contours_with_color(square_ragion, [contour])

    ''' get the mass center of  each ragion'''
    sqlited_squares = cv2_helper.Quadrilateral.split(square_contour, SUDOKU_COUNT, SUDOKU_COUNT)
    ragion_centers = map(cv2_helper.Contour.mass_center, sqlited_squares)
    # cv2_helper.Image.show_contours_with_color(square_ragion, [number_contours[0]])
    # cv2_helper.Image.show_points_with_color(square_ragion, [ragion_centers[18], cv2_helper.Contour.mass_center(number_contours[0])])
    # cv2_helper.Image.show_points_with_color(square_ragion, ragion_centers)

    ''' catalogue each number contour to ragion '''
    catalogue_contours(number_contours, ragion_centers)
    ''' join the number contours in same ragion '''
    ''' get the numbers' rect ragion'''

def catalogue_contours(contours, ragion_centers):
    def get_index(center):
        distances = map(lambda r: cv2.norm(center,r), ragion_centers)
        return distances.index(min(distances))

    contour_centers = map(cv2_helper.Contour.mass_center, contours)
    return map(get_index, contour_centers)



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
        show_square('../resource/example_pics/sample14.dataset.jpg')
        # for i in range(1,15):
        #     image_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        #     show_square(image_path)

        pass