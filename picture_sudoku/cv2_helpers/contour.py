import cv2
import numpy

from rect import Rect
from picture_sudoku.helpers import list_helper


class Contour(object):
    @staticmethod
    def mass_center(contour):
        '''
            Contour.mass_center
        '''
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return (1, 1)
        return ( moments['m10']/moments['m00'], moments['m01']/moments['m00'])

    @staticmethod
    def get_rect_ragion(contour, the_image):
        '''
            Contour.get_rect_ragion
        '''
        return Rect.get_ragion(cv2.boundingRect(contour), the_image)


    @staticmethod
    def vertices_by_min_area(contour):
        '''
            Contour.vertices_by_min_area
        '''
        ''' another way to get the rect of a contour
            just like boundingRect
        '''
        rect = cv2.minAreaRect(contour)
        points = cv2.cv.BoxPoints(rect)
        points = numpy.int0(points)
        return points

    @staticmethod
    def check_beyond_borders(contour, the_shape):
        '''
            Contour.check_beyond_borders
            width = end_x - start_x + 1, so x must <= width - 1
        '''
        height, width = the_shape
        # deep copy 
        changed_contour = numpy.empty_like(contour)
        changed_contour[:] = contour

        for point_arr in changed_contour:
            x, y = point_arr[0]
            point_arr[0,0] = list_helper.adjust_in_range(x, 0, width - 1)
            point_arr[0,1] = list_helper.adjust_in_range(y, 0, height - 1)

        return changed_contour

    @staticmethod
    def get_shape(contour):
        '''
            Contour.get_shape
            width = end_x + 1
        '''
        rect = cv2.boundingRect(contour)
        x, y, width, height = rect
        x = max(x,0)
        y = max(y,0)
        return (height+y, width+x)


if __name__ == '__main__':
    from minitest import *
    from image import Image
    from display import Display

    ORIGINAL_IMAGE_NAME = '../../resource/example_pics/sample01.dataset.jpg'
    gray_image = cv2.imread(ORIGINAL_IMAGE_NAME, 0)

    with test(Contour.mass_center):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Contour.mass_center(contour).must_equal(
            (215.54283010213263, 405.8626870351339))

    with test(Contour.get_rect_ragion):
        contour = numpy.array([[[ 171,  21]],
                               [[  18,  26]],
                               [[  25, 216]],
                               [[ 175, 212]]])
        '''
            notice the result will be different with different picture.
        '''
        part_image1 = Contour.get_rect_ragion(contour, gray_image)
        part_image1.shape.must_equal((196, 158))
        part_image1[0,0].must_equal(gray_image[21,18])
        # Display.image(part_image1)

        rect = (18, 21, 158, 196)
        cv2.boundingRect(contour).must_equal(rect)
        part_image2 = Rect.get_ragion(rect, gray_image)
        part_image2.shape.must_equal((196, 158))
        part_image2[0,0].must_equal(gray_image[21,18])
        ''' uncomment the below, you can see the consequence in a picture. '''
        # Display.image(part_image1)    

    with test(Contour.check_beyond_borders):
        contour = numpy. array(
              [[[ -1, 116]],
               [[334, 202]],
               [[318, 457]],
               [[ 74, 460]]])
        height, width = 800, 600
        the_image = Image.generate_mask((height, width))
        changed_contour = Contour.check_beyond_borders(contour, the_image.shape)
        changed_contour[0,0,0].must_equal(0)
        # Display.contours(the_image, [contour])
        # Display.contours(the_image, [changed_contour])

        the_image = Image.generate_mask((460, 333))
        changed_contour = Contour.check_beyond_borders(contour, the_image.shape)
        changed_contour[1,0,0].must_equal(332)
        changed_contour[3,0,1].must_equal(459)
        # Display.contours(the_image, [contour])
        # Display.contours(the_image, [changed_contour])

    with test(Contour.get_shape):
        contour = numpy. array(
              [[[ -1, 116]],
               [[334, 202]],
               [[88, 66]],
               [[318, 457]],
               [[ 74, 460]]])
        # notice the width not 335 but 336, because the -1
        Contour.get_shape(contour).must_equal((461, 336))
