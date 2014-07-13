import cv2
import numpy
import operator

from picture_sudoku.helpers import list_helper

class PolarLines(object):
    ''' 
        for the lines in Polar coordinate system
        rho, theta = line 
    '''

    @staticmethod
    def find_suitable_lines(the_image):

        threshold = 100
        condition = True

        while condition:
            lines = cv2.HoughLines(the_image, rho=1, theta=numpy.pi/180, threshold= threshold)
            lines = lines[0]

            threshold += 10
            condition = len(lines) > 40 and threshold < 400

        # (len(lines), threshold-10).ppl()
        return lines

    @staticmethod
    def catalogue_lines(lines, accuracy_pixs):
        ''' accuracy_pixs is used to be judge the line's class '''
        return list_helper.catalogue_list_list(lines, 0, accuracy_pixs)
        # sorted_lines = sorted(lines, key=operator.itemgetter(0))

        # pre_rho = sorted_lines[0][0]
        # catalogued_lines = [ [sorted_lines[0]] ]
        # cur_index = 0
        # for cur_line  in sorted_lines[1::]:
        #     rho, theta = cur_line
        #     if pre_rho + accuracy_pixs > rho:
        #         catalogued_lines[cur_index].append(cur_line)
        #     else:
        #         cur_index += 1
        #         catalogued_lines  += [[cur_line]]
        #     pre_rho = rho
        # return catalogued_lines

    # @staticmethod
    # def remove_minority_by_theta(lines):
    #     ''' delta equals 20 degree '''
    #     delta = 0.348
    #     standard_theta = lines[0][1]
    #     result_lines = [ [lines[0]] ]
    #     for cur_line  in lines[1::]:
    #         rho, theta = line
    #         # if 

    @staticmethod
    def cal_mean_lines(catalogued_lines):
        ''' '''
        mean_lines = tuple(numpy.mean(lines, axis=0) 
            for lines in catalogued_lines)
        mean_lines = sorted(mean_lines, key=operator.itemgetter(0))
        return mean_lines

    @staticmethod
    def gen_middle_lines(first_line, last_line, count):
        step_line = (last_line - first_line) / (count + 1)
        return [first_line + step_line * (i+1) for i in range(count)]

    @staticmethod
    def fill_lost_lines(lines, line_count):
        ''' now, I ignore the situation which is lost the edge'''
        if len(lines) >= line_count:
            return lines
        first_line = lines[0]
        step = (lines[-1][0] - first_line[0]) / float(line_count-1)
        # step.ppl()
        all_lines = [first_line]
        pre_index = 0
        pre_line = first_line
        for cur_line in lines[1::]:
            distance = cur_line[0] - pre_line[0]
            index_delta = int(round( distance / step) - 1)
            if index_delta > 0:
                all_lines += PolarLines.gen_middle_lines(pre_line, cur_line, index_delta)
            all_lines.append(cur_line)
            pre_line = cur_line
            # don't change step now
        return all_lines

    @staticmethod
    def cal_intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        sin_t1 = numpy.sin(theta1)
        cos_t1 = numpy.cos(theta1)
        sin_t2 = numpy.sin(theta2)
        cos_t2 = numpy.cos(theta2)
        # (sin_t1, sin_t2, cos_t1, cos_t2).ppl()
        x_value = (sin_t1*rho2 - sin_t2*rho1) / (sin_t1*cos_t2 - sin_t2*cos_t1)
        if abs(sin_t1) < 0.00001:
            y_value = rho2 / sin_t2 - cos_t2 / sin_t2 * x_value
            # '11'.pl()
        else:
            y_value = rho1 / sin_t1 - cos_t1 / sin_t1 * x_value
            # '22'.pl()
        # (x_value, y_value).ppl()
        return (x_value, y_value)

    @staticmethod
    def cal_degree(line):
        return PolarLines.cal_degree_from_angle(line[1])

    @staticmethod
    def cal_angle_from_degree(the_degree):
        return the_degree * numpy.pi / 180

    @staticmethod
    def cal_degree_from_angle(the_angle):
        return (the_angle * 180/ numpy.pi)

    @staticmethod
    def create_from_rho_and_degree(rho, the_degree):
        # return (line[1] * 180/ numpy.pi)
        return numpy.array([ rho, PolarLines.cal_angle_from_degree(the_degree)])

    @staticmethod
    def adjust_to_positive_rho_line(line):
        rho, theta = line
        if rho < 0:
            rho = abs(rho)
            theta = theta - numpy.pi
        return numpy.array([rho, theta])

    @staticmethod
    def adjust_theta_when_rho_0_and_near_180(line):
        rho, theta = line
        # largar than 170 degree
        if theta > 2.967:
            theta =  theta - numpy.pi
            rho = 0 - rho
        # if rho ==0 and theta > 170 / 180 * numpy.pi:
        #     theta =  theta - numpy.pi
            # theta =  numpy.pi - theta
        return numpy.array([rho, theta])

if __name__ == '__main__':
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.cv2_helpers.image import Image
    from minitest import *
    inject(numpy.allclose, 'must_close')

    SUDOKU_SIZE = 9


    with test("PolarLines.catalogue_lines"):
        lines = [numpy.array([ 214.        ,    1.57079637]),
                 numpy.array([ 319.        ,    1.57079637]),
                 numpy.array([ 10.        ,   1.57079637]),
                 numpy.array([ 6.        ,  1.57079637]),
                 numpy.array([ 13.        ,   1.57079637]),
                 numpy.array([ 221.        ,    1.57079637]),
                 numpy.array([ 326.        ,    1.57079637]),
                 numpy.array([ 217.        ,    1.58824956]),
                 numpy.array([ 113.        ,    1.58824956]),
                 numpy.array([ 44.        ,   1.57079637]),
                 numpy.array([ 5.        ,  1.58824956]),
                 numpy.array([ 16.        ,   1.55334306]),
                 numpy.array([ 322.        ,    1.58824956]),
                 numpy.array([ 219.        ,    1.55334306]),
                 numpy.array([ 117.        ,    1.55334306]),
                 numpy.array([ 318.        ,    1.58824956]),
                 numpy.array([ 110.        ,    1.58824956]),
                 numpy.array([ 14.        ,   1.53588974]),
                 numpy.array([ 114.        ,    1.57079637]),
                 numpy.array([ 9.        ,  1.55334306]),
                 numpy.array([ 324.        ,    1.55334306]),
                 numpy.array([ 224.        ,    1.53588974]),
                 numpy.array([ 329.        ,    1.55334306]),
                 numpy.array([ 11.        ,   1.55334306]),
                 numpy.array([ 111.        ,    1.57079637])]
        accuracy_pixs = 330 / SUDOKU_SIZE *0.5 # 9
        # accuracy_pixs.ppl()
        catalogued_lines = PolarLines.catalogue_lines(lines, accuracy_pixs)
        catalogued_lines.size().must_equal(5)
        # sorted(catalogued_lines.keys()).must_equal([5.0, 44.0, 110.0, 214.0, 318.0])

    with test('PolarLines.cal_mean_lines'):
        mean_lines =[numpy.array([ 10.5      ,   1.5620697]),
                     numpy.array([ 44.        ,   1.57079637]),
                     numpy.array([ 113.        ,    1.57428698]),
                     numpy.array([ 219.        ,    1.56381502]),
                     numpy.array([ 323.        ,    1.57079633])]
        PolarLines.cal_mean_lines(catalogued_lines).must_equal(
            mean_lines, numpy.allclose)

    with test("PolarLines.gen_middle_lines"):
        mean_lines =[numpy.array([ 10.5      ,   1.5620697]),
                     numpy.array([ 44.        ,   1.57079637]),
                     numpy.array([ 113.        ,    1.57428698]),
                     numpy.array([ 219.        ,    1.56381502]),
                     numpy.array([ 323.        ,    1.57079633])]
        PolarLines.gen_middle_lines(mean_lines[3], mean_lines[4], 2).must_close(
            [numpy.array([ 253.66666667,    1.56614212]), numpy.array([ 288.33333333,    1.56846923])])

    with test("PolarLines.fill_lost_lines"):
        mean_lines =[numpy.array([ 10.5      ,   1.5620697]),
                     numpy.array([ 44.        ,   1.57079637]),
                     numpy.array([ 113.        ,    1.57428698]),
                     numpy.array([ 219.        ,    1.56381502]),
                     numpy.array([ 323.        ,    1.57079633])]
        all_lines = PolarLines.fill_lost_lines(mean_lines, SUDOKU_SIZE+1)
        # all_lines.ppl()
        all_lines.size().must_equal(SUDOKU_SIZE+1)

        mean_lines =[numpy.array([ 3.5       ,  1.57079637]),
                     numpy.array([ 102.54545593,    1.57555628]),
                     numpy.array([ 196.        ,    1.57603228]),
                     numpy.array([ 284.8888855 ,    1.57661402])]

        all_lines = PolarLines.fill_lost_lines(mean_lines, SUDOKU_SIZE+1)
        # all_lines.ppl()
        all_lines.size().must_equal(SUDOKU_SIZE+1)


    with test("PolarLines.cal_intersection"):
        line1 = numpy.array([ 1      ,   0])
        line2 = numpy.array([ 2      ,   numpy.pi/2])
        point = PolarLines.cal_intersection(line1, line2)
        point.must_close((1,2))

        line1 = numpy.array([ 1      ,   0])
        line2 = numpy.array([ 2      ,   -numpy.pi/2])
        point = PolarLines.cal_intersection(line1, line2)
        point.must_close((1,-2))
        
        line1 = numpy.array([ 5.75      , -0.08726659])
        line2 = numpy.array([ 8.75      ,  1.56643295])
        point = PolarLines.cal_intersection(line1, line2)
        point.must_close((6.5350032, 8.7215652))

        line1 = numpy.array([ 1.14250000e+02,  -1.95577741e-08], dtype=numpy.float32)
        line2 = numpy.array([ 79.25      ,   1.56643295], dtype=numpy.float32)
        point = PolarLines.cal_intersection(line1, line2)
        point.must_close((114.25, 78.752235466022668))

    with test(PolarLines.adjust_to_positive_rho_line):
        the_line = PolarLines.create_from_rho_and_degree(10, 178)
        PolarLines.adjust_to_positive_rho_line(the_line).must_close(
            PolarLines.create_from_rho_and_degree(10, 178))

        the_line = PolarLines.create_from_rho_and_degree(-10, 178)
        PolarLines.adjust_to_positive_rho_line(the_line).must_close(
            PolarLines.create_from_rho_and_degree(10, -2))

    with test(PolarLines.adjust_theta_when_rho_0_and_near_180):
        the_line = PolarLines.create_from_rho_and_degree(0, 178)
        PolarLines.adjust_theta_when_rho_0_and_near_180(the_line).must_close(
            PolarLines.create_from_rho_and_degree(0, -2))
        the_line = numpy.array([  340,   0.06] )
        PolarLines.adjust_theta_when_rho_0_and_near_180(the_line).must_close(
            the_line)

        # the_image = Image.generate_mask((800,600))
        # Display.polar_lines(the_image, [the_line])



    with test("show line degree"):
        # 2,3,5 same, 1,4 same
        line1 = PolarLines.create_from_rho_and_degree(10, 178)
        line2 = PolarLines.create_from_rho_and_degree(10, -2)
        line3 = PolarLines.create_from_rho_and_degree(-10, -182)
        line4 = PolarLines.create_from_rho_and_degree(-10, -2)
        line5 = PolarLines.create_from_rho_and_degree(10, 358)
        the_image = Image.generate_mask((800,600))
        # # # Display.polar_lines(the_image, [line2,line3, line4])
        # Display.polar_lines(the_image, [line1, line2])
        # Display.polar_lines(the_image, [line2,])


        lines = [numpy.array([ 1.        , -0.12217298]),
                 numpy.array([ 1.        ,  3.01941967]),
                 numpy.array([ 3.        ,  3.00196624]),
                 numpy.array([ 5.        ,  3.00196624]),
                 numpy.array([ 7.        ,  3.00196624])]
        # Display.polar_lines(the_image, lines[:1])
        # Display.polar_lines(the_image, lines[:1])

