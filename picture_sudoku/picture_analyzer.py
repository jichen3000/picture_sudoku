import numpy
import cv2

from picture_sudoku.helpers import cv2_helper
from picture_sudoku.helpers import numpy_helper


BLACK = cv2_helper.BLACK
WHITE = cv2_helper.WHITE

IMG_SIZE = 32
'''
    notice, the point in the pic_arr, like [478, 128], 
    the first one is the column number, 
    the second one is the row number.
    Don't get them reversely.
'''
def find_sudoku_number_binary_arr(gray_pic_arr):
    '''
        Find all numbers from a picture in which there is a soduku puzzle.
        The number form is a binary numpy.array in which number parts are 1, 
        the others are 0.
    '''
    '''
        notice: the threshold_value is the key, if it directly impact the binary matrix.
    '''
    threshed_pic_array = cv2_helper.threshold_white_with_mean_percent(gray_pic_arr)
    # not_use.pp()

    square = find_max_square(threshed_pic_array)

    number_rects = cv2_helper.cal_split_ragion_rects(square, 9, 9)
    # cv2_helper.show_rects_in_pic(gray_pic_arr, number_rects)


    binary_pic = numpy_helper.transfer_values_quickly(threshed_pic_array, {BLACK:0, WHITE:1})
    number_binary_ragions = map(lambda c: cv2_helper.get_rect_ragion_with_rect(binary_pic, c),
        number_rects)

    number_binary_ragions = map(remove_border, number_binary_ragions)

    non_empty_indexs, number_binary_ragions = get_nonzero_ragions_and_indexs(number_binary_ragions)
    # indexs.pp()

    # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

    number_binary_ragions = map(remove_margin, number_binary_ragions)

    number_binary_ragions = map(enlarge, number_binary_ragions)
    # show_pic(threshed_pic_array)
    # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

    return number_binary_ragions

def remove_border(pic_array):
    return cv2_helper.clip_array_by_percent(pic_array, percent=0.15)

def get_nonzero_ragions_and_indexs(ragions):
    index_ragion_list = [(i, ragion) for i, ragion in enumerate(ragions) 
            if cv2_helper.is_not_empty_pic(ragion)]
    return zip(*index_ragion_list)



def remove_margin(pic_array):
    new_rect = cv2_helper.cal_nonzero_rect_as_pic_ratio(pic_array)
    return cv2_helper.get_rect_ragion_with_rect(pic_array, new_rect)
    
def enlarge(pic_array):
    return cv2.resize(pic_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC);


def find_max_square(threshed_pic_array):
    squares = cv2_helper.find_contours(threshed_pic_array, is_almost_square)
    square_perimeter_arr = [cv2.arcLength(i,True) for i in squares]
    return squares[square_perimeter_arr.index(max(square_perimeter_arr))]


def is_almost_square(contour, accuracy=0.001):
    '''
        The accuracy is the key, and cannot larger than 0.001
    '''    
    if len(contour)!=4:
        return False
    perimeter = cv2.arcLength(contour, True)
    area_from_perimeter = (perimeter / 4) ** 2
    real_area = cv2.contourArea(contour)
    # area_from_perimeter.pp()
    # real_area.pp()
    if (1-accuracy) * area_from_perimeter < real_area < (1+accuracy) * area_from_perimeter:
        return True
    return False


def clip_array_by_fixed_size(pic_array, fixed_height=32, fixed_width=32, delta_start_y=3):
    height, width = pic_array.shape
    start_y = int((height - fixed_height)/2)-delta_start_y
    start_x = int((width - fixed_width)/2)
    return pic_array[start_y:start_y+fixed_height, start_x:start_x+fixed_width]


if __name__ == '__main__':
    from minitest import *

    def show_pic(pic_arr):
        cv2.imshow('pic', pic_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ORIGINAL_IMAGE_NAME = '../resource/example_pics/original.jpg'
    # 0 is black, white is 255
    # large is brighter, less is darker.

    gray_arr = cv2.imread(ORIGINAL_IMAGE_NAME, 0)
    gray_area_arr = gray_arr[400:1100,50:700]
    color_arr = cv2.imread(ORIGINAL_IMAGE_NAME)
    color_area_arr = color_arr[400:1100,50:700]


    with test("is_almost_square"):
        contour = numpy.array([[[ 671,  421]],
                               [[  78,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]])
        is_almost_square(contour).must_equal(True)
        contour = numpy.array([[[ 671,  421]],
                               [[  128,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]])
        is_almost_square(contour).must_equal(False)


    with test("find_max_square"):
        current_pic_arr = cv2_helper.threshold_white_with_mean_percent(gray_arr)
        max_square = find_max_square(current_pic_arr)
        max_square.must_equal(numpy.array([[[ 671,  421]],
                               [[  78,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]]), numpy.allclose)
        (cv2.arcLength(max_square,True) > 2300).must_equal(True)
        pass



    # with test("show max square in full pic"):
    #     current_pic_arr = color_arr
    #     cv2.drawContours(current_pic_arr,[max_square],-1,(0,255,255),1)
    #     show_pic(current_pic_arr)

    # with test("show max square in area pic"):
    #     current_pic_arr = cv2_helper.threshold_white_with_mean_percent(gray_area_arr)
    #     # current_pic_arr = color_area_arr
    #     area_max_square = find_max_square(current_pic_arr)
    #     cv2.drawContours(current_pic_arr,[area_max_square],-1,(0,255,255),1)
    #     show_pic(current_pic_arr)


    with test("find_sudoku_number_binary_arr"):
        number_binary_arr = find_sudoku_number_binary_arr(gray_arr)
        number_5 = number_binary_arr[0]
        black_count = numpy.count_nonzero(number_5)
        white_count = numpy.count_nonzero(1-number_5)
        row_count, col_count = number_5.shape
        (row_count*col_count).must_equal(black_count+white_count)
        black_count.must_equal(364)
        # number_5 = clip_array_by_fixed_size(number_5,delta_start_y=-5)
        numpy.savetxt("test5.dataset",number_5,fmt="%d", delimiter='')
