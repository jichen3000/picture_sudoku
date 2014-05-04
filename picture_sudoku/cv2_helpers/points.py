import cv2
import numpy

class Points(object):
    @staticmethod
    def get_quadrilaterals(the_points, count_in_col, count_in_row):
        '''
            Points.get_quadrilaterals
            Get the contours which are quadrilaterals.
        '''
        all_points = the_points.reshape((count_in_col+1, count_in_row+1,2))
        quadrilaterals = [Points.to_contour(
                    [all_points[x_index,    y_index], 
                     all_points[x_index,    y_index+1], 
                     all_points[x_index+1,    y_index+1], 
                     all_points[x_index+1,    y_index]]) 
                for x_index in range(count_in_col) for y_index in range(count_in_row)]
        return quadrilaterals


    @staticmethod
    def cal_step_points(two_endpoints_in_line, count):
        '''
            Points.cal_step_points
            two_endpoints_in_line are the two point of a line.
            count will be the points on that line between the two endpoints.
        '''
        step_distance = numpy.float32(two_endpoints_in_line[-1] - two_endpoints_in_line[0]) / (count-1)
        points_list = [(two_endpoints_in_line[0] + index * step_distance)
                for index in range(count)]
        return numpy.array(numpy.int32(points_list)).reshape((count,2))
    
    @staticmethod
    def cal_internal_points(vertices, row_count, col_count):
        '''
            Points.cal_internal_points
            The point's order will be from left to right then top to bottom.
        '''
        left_points = Points.cal_step_points(vertices[0:2], row_count)
        rifht_points = Points.cal_step_points(vertices[::-1][0:2], row_count)
        internal_points = [] 
        for left_point, right_point in zip(left_points, rifht_points):
            # (left_point, right_point).pp()
            endpoints = numpy.append(left_point, right_point).reshape(2,2)
            internal_points.append(Points.cal_step_points(endpoints, col_count))

        return numpy.array(internal_points).reshape((col_count)*(row_count),2)

    @staticmethod
    def to_contour(points):
        '''
            Points.to_contour
            just support 4 points
        '''
        return numpy.array(points).reshape((4,1,2))

    @staticmethod
    def cal_line_slope(point1, point2):
        '''
            Points.cal_line_slope
            calculate the line's slope which go through the two points
        '''
        x1, y1 = point1
        x2, y2 = point2
        if (x1 - x2) == 0 and (y1 - y2) == 0:
            raise Exception('These two points are same.')
        if (x1 - x2) == 0:
            return float('inf')
        return (float(y1 - y2) / (x1 - x2))

if __name__ == '__main__':
    from minitest import *
    with test("Points.cal_line_slope"):
        points = numpy.array([[-1,1],[1,0]])
        Points.cal_line_slope(points[0], points[1]).must_equal(-0.5)
        points = ([1,1],[1,-1])
        Points.cal_line_slope(points[0], points[1]).must_equal(float('inf'))

    with test("Points.cal_step_points"):
        endpoints = numpy.array([[153,  13],
            [187, 640]], dtype=numpy.int32)
        Points.cal_step_points(endpoints, 10).must_equal(
            numpy.array(  [[153,  13],
                           [156,  82],
                           [160, 152],
                           [164, 222],
                           [168, 291],
                           [171, 361],
                           [175, 431],
                           [179, 500],
                           [183, 570],
                           [187, 640]]), numpy.allclose)
    with test("Points.cal_internal_points"):
        vertices = numpy.array(
              [[ 42, 249],
               [ 49, 583],
               [ 384, 569],
               [ 384, 225]], dtype=numpy.int32)
        points = Points.cal_internal_points(vertices, row_count=4, col_count=3)
        points.must_equal(numpy.array([[ 42, 249],
                                       [213, 237],
                                       [384, 225],
                                       [ 44, 360],
                                       [214, 349],
                                       [384, 339],
                                       [ 46, 471],
                                       [215, 462],
                                       [384, 454],
                                       [ 49, 583],
                                       [216, 576],
                                       [384, 569]], dtype=numpy.int32), numpy.allclose)
        ''' show '''
        # the_image = Image.generate_mask((600, 400))
        # Image.show_points_with_color(the_image, points)

