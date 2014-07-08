import cv2
import numpy

from rect import Rect
from ragion import Ragions
from picture_sudoku.helpers import numpy_helper

BLACK = 0
WHITE = 255

class Display(object):
    '''
        It's for showing the image.
        I cannot name it logically, I mean use a noun as the name, 
        and use the verb as the method name.
        Also the first parameter of method is not the display object.
    '''
    @staticmethod
    def image(the_image, image_name='image'):
        cv2.imshow(image_name, the_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def image_binary(the_image, image_name='image'):        
        Display.image(numpy_helper.transfer_values_quickly(
            the_image,{1:255}))
        
    @staticmethod
    def images(images, image_name='image'):
        dx = 0
        for index, cur_image in enumerate(images):
            window_name = image_name+str(index+1)
            cv2.imshow(window_name, cur_image)
            cv2.moveWindow(window_name, dx, 0)
            dx = cur_image.shape[1]
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def contours(the_image, contours, color=(0,255,255)):
        color_image = cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, contours, -1, color ,1)
        Display.image(color_image)

    @staticmethod
    def points(the_image, points, color=(0,255,0)):
        int_points = numpy.int32(points)
        color_image = cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)
        for point in int_points:
            cv2.circle(color_image,tuple(point),2,color,-1)
        Display.image(color_image)

    @staticmethod
    def rect(the_image, rect):
        rect_image = Rect.get_ragion(rect, the_image)
        Display.image(rect_image)


    @staticmethod
    def rects(the_image, rects, color=(0,255,255)):
        color_image = cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)
        contours = map(Rect.to_contour, rects)
        Display.contours(color_image, contours, color)

    @staticmethod
    def same_size_ragions(ragions, count_in_row=9, init_value=BLACK):
        '''
            Show the ragions which have the same size.
        '''
        the_image = Ragions.join_same_size(ragions, count_in_row, init_value)
        Display.image(the_image)

    @staticmethod
    def ragions(ragions, count_in_row=9, init_value=BLACK):
        '''
            Show the ragions which could have different size.
        '''
        same_size_ragions = Ragions.fill_to_same_size(ragions)
        count_in_row = min(len(ragions), count_in_row)
        Display.same_size_ragions(same_size_ragions, count_in_row, init_value)

    @staticmethod
    def ragions_binary(ragions, count_in_row=9, init_value=BLACK):
        '''
            Show the ragions which could have different size.
        '''
        transfered_ragions = [numpy_helper.transfer_values_quickly(ragion,{1:255}) 
            for ragion in ragions]
        Display.ragions(transfered_ragions)

    @staticmethod
    def polar_lines(the_image, lines):
        color_image = cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)
        for rho, theta in lines:
            cos_theta = numpy.cos(theta)
            sin_theta = numpy.sin(theta)
            x0 = cos_theta * rho
            y0 = sin_theta * rho
            point0 = (int(numpy.around(x0 + 1000*(- sin_theta))),  int(numpy.around(y0 + 1000*( cos_theta))))
            point1 = (int(numpy.around(x0 - 1000*(- sin_theta))),  int(numpy.around(y0 - 1000*( cos_theta))))
            cv2.line(color_image, point0, point1, (0,0,255), thickness=2)
        Display.image(color_image)

if __name__ == '__main__':
    from minitest import *

