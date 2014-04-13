import cv2
import numpy
import operator

from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

from picture_sudoku.helpers.cv2_helpers.image import Image
# from picture_sudoku.helpers.cv2_helpers.ragion import Ragion
# from picture_sudoku.helpers.cv2_helpers.ragion import Ragions
from picture_sudoku.helpers.cv2_helpers.rect import Rect
# from picture_sudoku.helpers.cv2_helpers.contour import Contour
# from picture_sudoku.helpers.cv2_helpers.quadrilateral import Quadrilateral
# from picture_sudoku.helpers.cv2_helpers.points import Points


def get_nonzero_rect(the_ragion):
    ''' 
        It will get the rect which has the nonzero in the center and can avoid getting the border.
        But how to choise the center rect is quite important, and sometimes, it will get blank line.
    '''
    ragion_height, ragion_width = the_ragion.shape
    # the_ragion.shape.ppl()

    centroid = Image.centroid(the_ragion)
    # centroid.ppl()
    nonzero_indexs = numpy.nonzero(the_ragion)
    nonzero_points = zip(*nonzero_indexs)
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
        # Display.image_rect(the_ragion, cur_rect)

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
