import cv2
import numpy

from contour import Contour
from points import Points

class Quadrilateral(Contour):
    @staticmethod
    def vertices(quadrilateral):
        '''
            Quadrilateral.vertices
            It will adjust points to top_left, bottom_left, bottom_right, top_right.
        '''
        points = quadrilateral.reshape((4,2))
        results = points.copy()
        x_mean, y_mean = numpy.mean(points, axis=0)
        for point in points:
            if point[0] < x_mean and point[1] < y_mean:
                results[0] = point
            if point[0] < x_mean and point[1] > y_mean:
                results[1] = point
            if point[0] > x_mean and point[1] > y_mean:
                results[2] = point
            if point[0] > x_mean and point[1] < y_mean:
                results[3] = point
        return results

    @staticmethod
    def centroid(quadrilateral):
        '''
            Quadrilateral.centroid 
        '''
        points = quadrilateral.reshape((4,2))
        return numpy.mean(points, axis=0)


    @staticmethod
    def split(quadrilateral,  count_in_row, count_in_col):
        '''
            Quadrilateral.split
            Split to many samll ones.
        '''
        vertices = Quadrilateral.vertices(quadrilateral)
        the_points = Points.cal_internal_points(vertices, count_in_col+1, count_in_row+1)
        return Points.get_quadrilaterals(the_points, count_in_col, count_in_row)

    @staticmethod
    def enlarge(quadrilateral, percent=0.05):
        '''
            Quadrilateral.enlarge
        '''
        x, y, width, height = cv2.boundingRect(quadrilateral)
        x_step = int(width * percent)
        y_step = int(height * percent)
        top_left, bottom_left, bottom_right, top_right = \
                Quadrilateral.vertices(quadrilateral)
        top_left=(top_left[0]-x_step, top_left[1]-y_step)
        top_right=(top_right[0]+x_step, top_right[1]-y_step)
        bottom_right=(bottom_right[0]+x_step, bottom_right[1]+y_step)
        bottom_left=(bottom_left[0]-x_step, bottom_left[1]+y_step)
        return Points.to_contour((top_left, top_right, bottom_right, bottom_left))

if __name__ == '__main__':
    from minitest import *

    with test("Quadrilateral.centroid"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Quadrilateral.centroid(contour).must_equal(
            numpy.array(
              [ 214.75,  406.5 ]), numpy.allclose)


    with test("Quadrilateral.split"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        contours = Quadrilateral.split(contour,2,3)
        contours.size().must_equal(6)
        contours[1].must_equal(numpy.array(
              [[[213, 237]],
               [[384, 225]],
               [[384, 339]],
               [[214, 349]]]), numpy.allclose)

        ''' show '''
        # the_image = Image.generate_mask((600, 400))
        # Image.show_contours_with_color(the_image, contours)
        # contours[1:2].pp()
        # Image.show_contours_with_color(the_image, contours[1:2])

    with test("Quadrilateral.enlarge"):
        quadrilateral = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        percent = 0.05
        enlarged_quadrilateral = Quadrilateral.enlarge(quadrilateral, percent)
        (cv2.arcLength(enlarged_quadrilateral, True) > cv2.arcLength(quadrilateral, True) * (1+percent*2)
            ).must_equal(True)

        ''' show '''
        # the_image = Image.generate_mask((700, 500))
        # Image.show_contours_with_color(the_image, [quadrilateral, enlarged_quadrilateral])

    with test("Quadrilateral.vertices"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Quadrilateral.vertices(contour).must_equal(numpy.array(
              [[ 42, 249],
               [ 49, 583],
               [ 384, 569],
               [ 384, 225]]), numpy.allclose)

