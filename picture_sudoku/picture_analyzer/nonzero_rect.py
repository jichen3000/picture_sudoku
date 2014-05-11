import cv2
import numpy
import operator

from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

from picture_sudoku.cv2_helpers.image import Image
# from picture_sudoku.cv2_helpers.ragion import Ragion
# from picture_sudoku.cv2_helpers.ragion import Ragions
from picture_sudoku.cv2_helpers.rect import Rect
# from picture_sudoku.cv2_helpers.contour import Contour
# from picture_sudoku.cv2_helpers.quadrilateral import Quadrilateral
# from picture_sudoku.cv2_helpers.points import Points

__all__ = ['analyze_from_center']

def analyze_from_center(the_ragion):
    first_rect = generate_first_rect(the_ragion)
    nonzeros = the_ragion.nonzero()

    nonzeros_in_rect = filter_nonzero_in_rect(nonzeros, first_rect)
    if not has_enough_nonzeros(nonzeros_in_rect, the_ragion.shape):
        return None

    nonzero_dicts = organize_nonzeros(nonzeros)
    real_rect = cal_smallest_rect(nonzeros_in_rect)
    if is_on_rect_borders(nonzero_dicts, real_rect):
        real_rect = enlarge_search_nonzero_rect(nonzero_dicts, real_rect)
        # sometime, the rect will just be the whole ragion,
        # in this case, it will be considered as None
        if real_rect[0] < 0 or real_rect[1] < 0:
            return None
    # else:
    #     real_rect = cal_smallest_rect(nonzeros_in_rect)
    return real_rect

def has_enough_nonzeros(nonzeros, the_shape):
    return (float(nonzeros[0].size) / (the_shape[0] * the_shape[1]) ) > 0.01

def cal_smallest_rect(nonzeros):
    y_indexs, x_indexs = nonzeros
    return (min(x_indexs), min(y_indexs), 
        max(x_indexs)-min(x_indexs)+1, max(y_indexs)-min(y_indexs)+1)


def filter_nonzero_in_rect(nonzeros, the_rect):
    result_ys = []
    result_xs = []
    start_x, start_y, width, height = the_rect
    # zip(nonzeros[0], nonzeros[1]).ppl()
    for y, x in zip(nonzeros[0], nonzeros[1]):
        if start_x <= x < start_x + width and start_y <= y < start_y + height :
            result_ys.append(y)
            result_xs.append(x)
    return (numpy.array(result_ys), numpy.array(result_xs))

def organize_nonzeros(nonzeros):
    points = zip(nonzeros[0], nonzeros[1])
    y_dict = {}
    x_dict = {}
    for y,x in points:
        if y in y_dict.keys():
            y_dict[y].append(x)
        else:
            y_dict[y] = [x]
        if x in x_dict.keys():
            x_dict[x].append(y)
        else:
            x_dict[x] = [y]
    return (y_dict, x_dict)

def enlarge_rect(the_rect, border_flags):
    left, top, width, height = the_rect
    non_left, non_top, non_right, non_bottom = border_flags
    if non_left:
        left -= 1
        width += 1
    if non_top:
        top -= 1
        height += 1
    if non_right:
        width += 1
    if non_bottom:
        height += 1
    return (left, top, width, height)

def generate_first_rect(the_ragion):
    ragion_height, ragion_width = the_ragion.shape

    centroid = Image.centroid(the_ragion)
    single_width = int(round(ragion_width*0.18))
    single_height = int(round(ragion_height*0.18))
    # (single_width, single_height).ppl()
    return Rect.cal_center_rect(centroid, single_width, single_width, single_height, single_height)


def enlarge_search_nonzero_rect(nonzero_dicts, first_rect):
    '''
        Search the continue nonzero points in a rect from a first rect.
    '''
    border_flags = is_on_rect_borders(nonzero_dicts, first_rect)
    result_rect = first_rect
    while border_flags:
        result_rect = enlarge_rect(result_rect, border_flags)
        border_flags = is_on_rect_borders(nonzero_dicts, result_rect)
    return result_rect


def is_on_rect_borders(nonzero_dicts, the_rect):
    '''
        Check whether or not there are nonzero point on the border of rect.
        If there is not any nonzero point on the border of rect, it returns False,
        otherwise, it returns the four bool value of the borders.
    '''
    left_x, top_y, width, height = the_rect
    right_x, bottom_y = left_x + width - 1, top_y + height - 1
    y_dict, x_dict = nonzero_dicts
    # non_left, non_right, non_top, non_bottom = False, False, False, False
    def check(the_dict, cur_value, low, high):
        if cur_value in the_dict.keys() :
            # the_dict[cur_value].ppl()
            # (cur_value,low,high).ppl()
            for the_item in the_dict[cur_value]:
                if low <= the_item <= high:
                    return True
        return False
    non_left = check(x_dict, left_x, top_y, bottom_y)
    non_top = check(y_dict, top_y, left_x, right_x)
    non_right = check(x_dict, right_x, top_y, bottom_y)
    non_bottom = check(y_dict, bottom_y, left_x, right_x)

    if not any((non_left, non_top, non_right, non_bottom)):
        return False
    else:
        return (non_left, non_top, non_right, non_bottom)

def center_rect_enlarge_search(the_ragion):
    '''
        it has been deprecated, since it will take more time and 
        cannot get the whole rect in some case.
    '''
    ragion_height, ragion_width = the_ragion.shape

    centroid = Image.centroid(the_ragion)
    single_width = int(round(ragion_width*0.18))
    single_height = int(round(ragion_height*0.18))
    # (single_width, single_height).ppl()
    left_x, top_y, edge_width, edge_height = \
            Rect.cal_center_rect(centroid, single_width, single_width, single_height, single_height)
    right_x = left_x + edge_width - 1
    bottom_y = top_y + edge_height - 1
    # (left_x, top_y, edge_width, edge_height).ppl()

    top_nonzero = True
    bottom_nonzero = True
    left_nonzero = True
    right_nonzero = True
    while(top_nonzero or bottom_nonzero or left_nonzero or right_nonzero):
        # (left_x, top_y, right_x, bottom_y).ppl()
        # (left_nonzero, top_nonzero, right_nonzero, bottom_nonzero).ppl()
        # cur_rect = Rect.create(left_x, top_y, right_x, bottom_y)
        # Display.rect(the_ragion, cur_rect)

        if top_nonzero:
            top_nonzero = Rect.has_nonzero((left_x, top_y, right_x-left_x+1, 1),the_ragion)
            if top_nonzero:
                top_y -= 1
            if top_y <= 0:
                top_nonzero = False
                top_y = -1
        if bottom_nonzero:
            bottom_nonzero = Rect.has_nonzero((left_x, bottom_y, right_x-left_x+1, 1),the_ragion)
            if bottom_nonzero:
                bottom_y += 1
            if bottom_y >= ragion_height-1:
                bottom_nonzero = False
                bottom_y = ragion_height
        if left_nonzero:
            left_nonzero = Rect.has_nonzero((left_x, top_y, 1, bottom_y-top_y+1),the_ragion)
            if left_nonzero:
                left_x -= 1
            if left_x <= 0:
                left_nonzero = False
                left_x = -1
        if right_nonzero:
            right_nonzero = Rect.has_nonzero((right_x, top_y, 1, bottom_y-top_y+1),the_ragion)
            if right_nonzero:
                right_x += 1
            if right_x >= ragion_width-1:
                right_nonzero = False
                right_x = ragion_width

    final_top = top_y + 1
    final_bottom = bottom_y - 1
    final_left = left_x + 1
    final_right = right_x - 1
    return Rect.create(final_left, final_top, final_right, final_bottom)


if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.helpers import numpy_helper

    inject(numpy.allclose, 'must_close')

    test_image_path = '../../resource/test/'
    image_14_07_path = test_image_path+'sample_14_07.dataset'
    image_14_07 = Image.read_from_number_file(image_14_07_path)
    image_14_07_255 = numpy_helper.transfer_values_quickly(image_14_07, {1:255})

    with test(analyze_from_center):
        rect_14_07 = analyze_from_center(image_14_07)
        rect_14_07.must_equal((14, 10, 28, 41))
        # Display.rect(image_14_07_255, rect_14_07)

        image_01_03_path = test_image_path+'sample_01_03.dataset'
        image_01_03 = Image.read_from_number_file(image_01_03_path)
        image_01_03_255 = numpy_helper.transfer_values_quickly(image_01_03, {1:255})
        rect_01_03 = analyze_from_center(image_01_03)
        rect_01_03.must_equal((26, 25, 14, 19))
        # Display.rect(image_01_03_255, rect_01_03)

        image_02_null_path = test_image_path+'sample_02_null.dataset'
        image_02_null = Image.read_from_number_file(image_02_null_path)
        image_02_null_255 = numpy_helper.transfer_values_quickly(image_02_null, {1:255})
        rect_02_null = analyze_from_center(image_02_null)
        rect_02_null.must_equal(None)
        # Display.image(image_02_null_255)

        image_07_01_path = test_image_path+'sample_07_01.dataset'
        image_07_01 = Image.read_from_number_file(image_07_01_path)
        image_07_01_255 = numpy_helper.transfer_values_quickly(image_07_01, {1:255})
        rect_07_01 = analyze_from_center(image_07_01)
        rect_07_01.must_equal(None)
        # Display.image(image_07_01_255)

        image_13_05_path = test_image_path+'sample_13_05.dataset'
        image_13_05 = Image.read_from_number_file(image_13_05_path)
        image_13_05_255 = numpy_helper.transfer_values_quickly(image_13_05, {1:255})
        rect_13_05 = analyze_from_center(image_13_05)
        rect_13_05.must_equal((16, 16, 18, 25))
        # Display.image(image_13_05_255)
        # Display.rect(image_13_05_255, rect_13_05)


    with test(has_enough_nonzeros):
        first_rect = generate_first_rect(image_14_07)
        # first_rect.ppl()

        nonzeros = image_14_07.nonzero()
        nonzeros_in_rect = filter_nonzero_in_rect(nonzeros, first_rect)
        has_enough_nonzeros(nonzeros_in_rect, image_14_07.shape).must_true()


    with test(enlarge_rect):
        non_left, non_top, non_right, non_bottom = True, False, False, True
        enlarge_rect((1,2,3,4), (non_left, non_top, non_right, non_bottom)).must_equal(
            (0, 2, 4, 5))

    with test(enlarge_search_nonzero_rect):
        first_rect = generate_first_rect(image_14_07)
        # first_rect.ppl()
        nonzero_dicts = organize_nonzeros(image_14_07.nonzero())
        # nonzero_dicts.ppl()
        the_rect = enlarge_search_nonzero_rect(nonzero_dicts, first_rect)
        # the_rect.must_equal((19, 12, 27, 38))
        # Display.rect(image_14_07_255, first_rect)
        # Display.rect(image_14_07_255, the_rect)
        
        # import timeit
        # timeit.timeit("__main__.enlarge_search_nonzero_rect(__main__.nonzero_dicts, __main__.first_rect)", 
        #     setup="import __main__", number=100).ppl()

    with test(filter_nonzero_in_rect):
        # arr = numpy.array([])
        arr = numpy.array(
              [[ 0.,  0.,  1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  1.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.]])
        filter_nonzero_in_rect(arr.nonzero(), (1,1,2,3)).must_close(
            (numpy.array([1, 2, 3]), numpy.array([1, 2, 1])))

    with test(organize_nonzeros):
        arr = numpy.array(
              [[ 0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  1.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.]])
        organize_nonzeros(arr.nonzero()).must_equal(
            ({1: [1], 2: [2, 3], 3: [1]}, {1: [1, 3], 2: [2], 3: [2]}))

    with test(is_on_rect_borders):
        nonzero_dicts = ({1: [1], 2: [2, 3], 3: [1]}, {1: [1, 3], 2: [2], 3: [2]})
        is_on_rect_borders(nonzero_dicts, (1, 1, 4, 4)).must_equal((True, True, False, False))
        is_on_rect_borders(nonzero_dicts, (0, 0, 4, 5)).must_equal((False, False, True, False))
        is_on_rect_borders(nonzero_dicts, (0, 1, 5, 4)).must_equal((False, True, False, False))

        is_on_rect_borders(nonzero_dicts, (0, 0, 1, 1)).must_equal(False)
        is_on_rect_borders(nonzero_dicts, (-1, -1, 100, 100)).must_equal(False)


    with test(center_rect_enlarge_search):
        the_rect = center_rect_enlarge_search(image_14_07)
        the_rect.must_equal((19, 12, 19, 38))
        # Display.rect(image_14_07_255, the_rect)

        # import timeit
        # timeit.timeit("__main__.center_rect_enlarge_search(__main__.image_14_07)", 
        #     setup="import __main__", number=100).ppl()

