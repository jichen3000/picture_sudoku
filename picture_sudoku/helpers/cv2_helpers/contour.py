import cv2
import numpy

from rect import Rect

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

if __name__ == '__main__':
    from minitest import *
    from image import Image

    ORIGINAL_IMAGE_NAME = '../../../resource/example_pics/sample01.dataset.jpg'
    gray_image = cv2.imread(ORIGINAL_IMAGE_NAME, 0)

    with test("Contour.mass_center"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Contour.mass_center(contour).must_equal(
            (215.54283010213263, 405.8626870351339))

    with test("Contour.get_rect_ragion"):
        contour = numpy.array([[[ 171,  21]],
                               [[  18,  26]],
                               [[  25, 216]],
                               [[ 175, 212]]])
        '''
            notice the result will be different with different picture.
        '''
        part_image1 = Contour.get_rect_ragion(contour, gray_image)
        part_image1.shape.must_equal((196, 158))
        part_image1[0,0].must_equal(gray_image[18,21])

        rect = (18, 21, 158, 196)
        cv2.boundingRect(contour).must_equal(rect)
        part_image2 = Rect.get_ragion(rect, gray_image)
        part_image2.shape.must_equal((196, 158))
        part_image2[0,0].must_equal(gray_image[18,21])
        ''' uncomment the below, you can see the consequence in a picture. '''
        # Image.show(part_image1)    


