import cv2
import numpy

class Rect(object):
    '''
        x, y, width, height = rect
    '''
 

    @staticmethod
    def modify_to_ratio(rect, shape):
        '''
            Rect.modify_to_ratio
            This method will return the rect which has the same ratio 
            of row count and col count
        '''
        x, y, width, height = rect
        pic_height, pic_width= shape
        #height need to be enlarge
        if pic_height * width > pic_width * height:
            new_height = int((pic_height * width) / pic_width)
            delta = int( (new_height - height)/2 )
            new_y = y - delta
            if new_y < 0:
                new_y = 0
            return x, new_y, width, new_height
        elif pic_height * width < pic_width * height:
            new_width = int((pic_width * height) / pic_height)
            delta = int((new_width-width) / 2)
            new_x = x - delta
            if new_x < 0:
                new_x = 0
            return new_x, y, new_width, height
        return rect

    @staticmethod
    def split_to_rects(rect, split_x_num, split_y_num):
        '''
            Rect.split_to_rects
        '''
        x, y, width, height = rect
        x_step = int(width / split_x_num)
        y_step = int(height / split_y_num)
        result = tuple((x+i*x_step, y+j*y_step, x_step, y_step) 
            for j in range(split_y_num) for i in range(split_x_num))
        return result

    @staticmethod
    def adjust_to_minimum(rect):
        '''
            Rect.adjust_to_minimum
        '''
        x, y, width, height = rect
        min_width_or_height = min(width, height)

        return (x, y, min_width_or_height, min_width_or_height)

    @staticmethod
    def get_ragion(rect, the_image):
        '''
            Rect.get_ragion
        '''
        x, y, width, height = rect
        return the_image[y:y+height,x:x+width]

    @staticmethod
    def vertices(rect):
        '''
            Rect.vertices
        '''
        x, y, width, height = rect
        return numpy.array([(x,         y),
                            (x,         y+height-1),
                            (x+width-1, y+height-1),
                            (x+width-1, y)])

    @staticmethod
    def to_contour(rect):
        '''
            Rect.to_contour
        '''
        return Rect.vertices(rect).reshape((4,1,2))

    @staticmethod
    def has_nonzero(rect, ragion):
        '''
            Rect.has_nonzero
            Check whether or not the rect in the ragion has any value which is not zero.
        '''
        x, y, width, height = rect
        for cur_x in range(x, x+width):
            for cur_y in range(y, y+height):
                # (cur_x, cur_y).pp()
                if ragion[cur_y, cur_x] > 0:
                    return True
        return False

    @staticmethod
    def cal_center_rect(centroid, top_distance, bottom_distance, left_distance, right_distance):
        '''
            Rect.cal_center_rect
            Get the rect from centroid.
        '''
        c_x, c_y = centroid
        x = c_x - left_distance
        y = c_y - top_distance
        width = left_distance + right_distance + 1
        height = top_distance + bottom_distance + 1
        return (x, y, width, height)

    @staticmethod
    def create(left, top, right, bottom):
        return (left, top, right-left+1, bottom-top+1)


if __name__ == '__main__':
    from minitest import *

    with test("Rect.cal_center_rect"):
        centroid = (16,13)
        Rect.cal_center_rect(centroid, 4, 5, 6, 7).must_equal((10, 9, 14, 10))
        
    with test("Rect.modify_to_ratio"):
        cur_rect = (10, 9, 12, 18)
        Rect.modify_to_ratio(cur_rect, (32, 32)).must_equal(
            (7, 9, 18, 18))

    with test("Rect.split_to_rects"):
        contour = numpy.array(
              [[[ 16, 139]],
               [[ 16, 225]],
               [[478, 225]],
               [[478, 139]]])
        ragion_rects = Rect.split_to_rects(cv2.boundingRect(contour), 10, 1)
        ragion_contours = map(Rect.to_contour, ragion_rects)
        ragion_contours.size().must_equal(10)
        ''' uncomment the below, you can see the consequence in a picture. '''        
        # show_contours_in_pic(color_pic, ragion_contours)
        pass

    with test("Rect.to_contour"):
        contour = numpy.array([[[ 18,  21]],
                               [[ 18, 216]],
                               [[175, 216]],
                               [[175,  21]]])
        rect = (18, 21, 158, 196)
        Rect.to_contour(rect).must_equal(contour, numpy.allclose)

    with test("Rect.vertices"):
        rect = (18, 21, 158, 196)
        Rect.vertices(rect).must_equal(
            numpy.array([  [ 18,  21],
                           [ 18, 216],
                           [175, 216],
                           [175,  21]]), numpy.allclose)

    with test("Rect.has_nonzero"):
        zeros_arr = numpy.zeros((9,9))
        zeros_arr[5,5] = 1
        Rect.has_nonzero((1,5, 6, 1), zeros_arr).must_true()
        Rect.has_nonzero((1,6, 6, 1), zeros_arr).must_false()
