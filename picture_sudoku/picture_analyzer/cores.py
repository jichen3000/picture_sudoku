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


def get_nonzero_rect_new(the_ragion):
    # nonzero_indexs = numpy.nonzero(the_ragion)
    # nonzero_points = zip(*nonzero_indexs)
    # nonzero_points.ppl()

    first_rect = get_first_rect(the_ragion)

    # if not have_enough_nonzeros(the_ragion, first_rect):
    #     return None

    # continue_flag = True
    # while continue_flag:
    #     continue_flag = False
    #     for cur_lateral in ['top','bottom','left','right']:
    #         if found_nonzeros(cur_lateral, borders, the_ragion):
    #             continue_flag = True
    #             borders[cur_lateral] += 1
    #         check_over

    rect, point_on_border_flag = find_rect(the_ragion, first_rect)
    if point_on_border_flag:
        real_rect, point_on_border_flag = find_rect(the_ragion, rect)
    else:
        real_rect, nonzero_count = cal_nonzero_rect
        if not have_enough_nonzeros(real_rect, nonzero_count):
            return None
    
    return real_rect

def center_rect_enlarge_search(the_ragion):
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

def get_nonzero_rect(the_ragion):
    ''' 
        It will get the rect which has the nonzero in the center and can avoid getting the border.
        But how to choise the center rect is quite important, and sometimes, it will get blank line.
    '''
    ragion_height, ragion_width = the_ragion.shape
    # the_ragion.shape.ppl()

    centroid = Image.centroid(the_ragion)
    # centroid.ppl()
    # nonzero_indexs = numpy.nonzero(the_ragion)
    # nonzero_points = zip(*nonzero_indexs)
    # nonzero_points.ppl()

    single_width = int(round(ragion_width*0.18))
    single_height = int(round(ragion_height*0.18))
    # (single_width, single_height).ppl()
    original_rect = Rect.cal_center_rect(centroid, single_width, single_width, single_height, single_height)
    # original_rect.pl()
    left_x, top_y, edge_width, edge_height = \
            Rect.cal_center_rect(centroid, single_width, single_width, single_height, single_height)
    # if edge_width > 
    right_x = left_x + edge_width - 1
    bottom_y = top_y + edge_height - 1
    # (left_x, top_y, edge_width, edge_height).ppl()
    # top
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
            else:
                final_top = top_y + 1
            if top_y <= 0:
                top_nonzero = False
                top_y = 0
                final_top = 0
        if bottom_nonzero:
            bottom_nonzero = Rect.has_nonzero((left_x, bottom_y, right_x-left_x+1, 1),the_ragion)
            if bottom_nonzero:
                bottom_y += 1
            else:
                final_bottom = bottom_y - 1
            if bottom_y >= ragion_height-1:
                bottom_nonzero = False
                bottom_y = ragion_height-1
                final_bottom = ragion_height-1
        if left_nonzero:
            left_nonzero = Rect.has_nonzero((left_x, top_y, 1, bottom_y-top_y+1),the_ragion)
            if left_nonzero:
                left_x -= 1
            else:
                final_left = left_x + 1
            if left_x <= 0:
                left_nonzero = False
                left_x = 0
                final_left = 0
        if right_nonzero:
            right_nonzero = Rect.has_nonzero((right_x, top_y, 1, bottom_y-top_y+1),the_ragion)
            if right_nonzero:
                right_x += 1
            else:
                final_right = right_x - 1 
            if right_x >= ragion_width-1:
                right_nonzero = False
                right_x = ragion_width-1
                final_right = ragion_width-1


    nonzero_rect = Rect.create(final_left, final_top, final_right, final_bottom)
    # nonzero_rect = (final_left, final_top, final_right-final_left+1, final_bottom-final_top+1)
    # nonzero_rect.ppl()
    # (final_left, final_top, final_right, final_bottom).ppl()
    # Rect.create(final_left-1, final_top-1, final_right+1, final_bottom+1).ppl()
    # original_rect.ppl()
    if nonzero_rect == original_rect:
    # if Rect.create(final_left-1, final_top-1, final_right+1, final_bottom+1) == original_rect:
        # "NN".pl()
        return None
    else:
        return nonzero_rect
        # return Rect.get_ragion(nonzero_rect, the_ragion)

if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.helpers import numpy_helper

    with test("get_nonzero_rect"):
        image_01_03_path = '../../resource/test/sample_01_03.dataset'
        image_01_03 = Image.read_from_number_file(image_01_03_path)
        image_01_03_255 = numpy_helper.transfer_values_quickly(image_01_03, {1:255, 0:0})
        the_rect = get_nonzero_rect(image_01_03_255)
        # Display.image(image_01_03_255)
        # Display.rect(image_01_03_255, the_rect)

    with test("center_rect_enlarge_search"):
        image_14_07_path = '../../resource/test/sample_14_07.dataset'
        image_14_07 = Image.read_from_number_file(image_14_07_path)
        image_14_07_255 = numpy_helper.transfer_values_quickly(image_14_07, {1:255, 0:0})
        the_rect = center_rect_enlarge_search(image_14_07)
        the_rect.must_equal((19, 12, 27, 38))
        # Display.rect(image_14_07_255, the_rect)

    # with test("get_nonzero_rect_new"):
    #     image_01_03_path = '../../resource/test/sample_01_03.dataset'
    #     image_01_03 = Image.read_from_number_file(image_01_03_path)
    #     image_01_03_255 = numpy_helper.transfer_values_quickly(image_01_03, {1:255, 0:0})
    #     the_rect = get_nonzero_rect_new(image_01_03_255)
    #     Display.image(image_01_03)
    #     # Display.rect(image_01_03_255, the_rect)



