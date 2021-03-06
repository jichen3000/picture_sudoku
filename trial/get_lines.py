'''
    this method came from stackoverflow,
    and it cannot work very well for many pictures.
'''

import cv2
import numpy
import operator

from picture_sudoku.helpers import cv2_helper
from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

# large is brighter, less is darker.
BLACK = cv2_helper.BLACK
WHITE = cv2_helper.WHITE

IMG_SIZE = 32
SUDOKU_SIZE = 9
SMALL_COUNT = 3

def find_max_contour(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.0001):
    contours = cv2_helper.Image.find_contours(
        threshed_image, filter_func, accuracy_percent_with_perimeter)
    if len(contours) == 0:
        return None
    contour_area_arr = [cv2.contourArea(i) for i in contours]
    max_contour = contours[contour_area_arr.index(max(contour_area_arr))]
    return max_contour

def get_approximated_contour(contour, accuracy_percent_with_perimeter=0.0001):
    perimeter = cv2.arcLength(contour,True)
    # notice the approximation accuracy is the key, if it is 0.01, you will find 4
    # if it is larger than 0.8, you will get nothing at all.
    approximation_accuracy = accuracy_percent_with_perimeter*perimeter
    return cv2.approxPolyDP(contour,approximation_accuracy,True)


def convert_to_square(contour):
    for i in reversed(range(1,10)):
        result = get_approximated_contour(contour, 0.01*i)
        # i.p()
        # len(result).p()
        if len(result)==4:
            return result
    return None


def cal_horizontal_max_angle(square_vertices):
    ''' it could be used in the filter lines'''
    square_vertices.pp()

    top_line_slope = cv2_helper.Points.cal_line_slope(square_vertices[0], square_vertices[3])
    top_line_angle = numpy.arctan(top_line_slope) * 180 / numpy.pi
    # square_vertices.p()
    bottom_line_slope = cv2_helper.Points.cal_line_slope(square_vertices[1], square_vertices[2])
    bottom_line_angle = numpy.arctan(bottom_line_slope) * 180 / numpy.pi
    (top_line_angle, bottom_line_angle).pp()
    max_angle = max(abs(top_line_angle), abs(bottom_line_angle))
    max_angle.p()
    return max_angle



def find_vertical_lines(gray_pic):
    gray_pic = cv2.GaussianBlur(gray_pic,ksize=(5,5), sigmaX=0)


    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

    dx = cv2.Sobel(gray_pic,cv2.CV_16S,2,0)
    # convert from dtype=int16 to dtype=uint8
    dx = cv2.convertScaleAbs(dx)

    low_threshold, high_threshold = 40, 80

    close = cv2.Canny(dx, low_threshold, high_threshold)

    lines = PolarLines.find_suitable_lines(close)

    # show_lines(close, lines)


    # show_lines(close, lines)
    # lines.pp()

    def filter_line(line):
        rho, theta = line
        theta_degree = (theta * 180/ numpy.pi)
        return abs(theta_degree) < 10

    def to_positive_rho(line):
        if line[0] < 0:
            line[0] = abs(line[0])
            line[1] = line[1] - numpy.pi
        return line

    lines = map(to_positive_rho, lines)
    lines = filter(filter_line, lines)
    # show_lines(close, lines)
    # lines.pp()

    # lines.pp()
    accuracy_pixs = gray_pic.shape[1] / SUDOKU_SIZE *0.3 # 9
    all_lines = PolarLines.cal_all_lines(lines, accuracy_pixs, SUDOKU_SIZE+1)

    # show_lines(close, all_lines)

    return all_lines


def find_horizontal_lines(gray_pic):

    def filter_line(line):
        rho, theta = line
        theta_degree = (theta * 180/ numpy.pi) - 90
        return abs(theta_degree) < 10

    gray_pic = cv2.GaussianBlur(gray_pic,ksize=(5,5), sigmaX=0)

    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))

    dy = cv2.Sobel(gray_pic,cv2.CV_16S,0,2)
    # convert from dtype=int16 to dtype=uint8
    dy = cv2.convertScaleAbs(dy)

    low_threshold, high_threshold = 40, 80

    close = cv2.Canny(dy, low_threshold, high_threshold)


    lines = PolarLines.find_suitable_lines(close)


    lines = filter(filter_line, lines)
    # lines.pp()

    # show_lines(close, lines)

    accuracy_pixs = gray_pic.shape[0] / SUDOKU_SIZE *0.3 # 9
    line_count = SUDOKU_SIZE+1
    # all_lines = PolarLines.cal_all_lines(lines, accuracy_pixs, SUDOKU_SIZE+1)
    catalogued_lines = PolarLines.catalogue_lines(lines, accuracy_pixs)
    # accuracy_pixs.p()
    # catalogued_lines.pp()
    mean_lines = PolarLines.cal_mean_lines(catalogued_lines)
    all_lines = PolarLines.fill_lost_lines(mean_lines, line_count)

    # show_lines(close, all_lines)



    return all_lines



def show_lines(the_image, lines):
    color_image = cv2.cvtColor(the_image.copy(), cv2.COLOR_GRAY2BGR)
    for rho, theta in lines:
        # (rho, theta).pp()
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho
        point0 = (int(numpy.around(x0 + 1000*(- sin_theta))),  int(numpy.around(y0 + 1000*( cos_theta))))
        point1 = (int(numpy.around(x0 - 1000*(- sin_theta))),  int(numpy.around(y0 - 1000*( cos_theta))))
        cv2.line(color_image, point0, point1, (0,0,255), thickness=2)
    cv2_helper.Image.show(color_image)



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

        # (len(lines), threshold-10).pp()
        return lines

    @staticmethod
    def catalogue_lines(lines, accuracy_pixs):
        ''' accuracy_pixs is used to be judge the line's class '''
        sorted_lines = sorted(lines, key=operator.itemgetter(0))

        pre_rho = sorted_lines[0][0]
        catalogued_lines = [ [sorted_lines[0]] ]
        cur_index = 0
        for cur_line  in sorted_lines[1::]:
            rho, theta = cur_line
            if pre_rho + accuracy_pixs > rho:
                catalogued_lines[cur_index].append(cur_line)
            else:
                cur_index += 1
                catalogued_lines  += [[cur_line]]
            pre_rho = rho
        return catalogued_lines


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
        step = (lines[-1][0] - first_line[0]) / float(line_count)
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
    def cal_all_lines(lines, accuracy_pixs, line_count):
        catalogued_lines = PolarLines.catalogue_lines(lines, accuracy_pixs)
        mean_lines = PolarLines.cal_mean_lines(catalogued_lines)
        all_lines = PolarLines.fill_lost_lines(mean_lines, line_count)
        return all_lines


    @staticmethod
    def cal_intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        sin_t1 = numpy.sin(theta1)
        cos_t1 = numpy.cos(theta1)
        sin_t2 = numpy.sin(theta2)
        cos_t2 = numpy.cos(theta2)
        # (sin_t1, sin_t2, cos_t1, cos_t2).pp()
        x_value = (sin_t1*rho2 - sin_t2*rho1) / (sin_t1*cos_t2 - sin_t2*cos_t1)
        if abs(sin_t1) < 0.00001:
            y_value = rho2 / sin_t2 - cos_t2 / sin_t2 * x_value
            # '11'.p()
        else:
            y_value = rho1 / sin_t1 - cos_t1 / sin_t1 * x_value
            # '22'.p()
        # (x_value, y_value).pp()
        return (x_value, y_value)

    @staticmethod
    def cal_degree(line):
        return (line[1] * 180/ numpy.pi)

def remove_border(pic_array):
    return cv2_helper.clip_array_by_four_percent(pic_array, 
        top_percent=0.06, bottom_percent=0.06, left_percent=0.10, right_percent=0.10)
    # return cv2_helper.clip_array_by_percent(pic_array, percent=0.08)




def main(image_path):
    img = cv2.imread(image_path)
    # img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = cv2_helper.Image.resize_keeping_ratio_by_height(gray)
    # img = cv2_helper.Image.resize_keeping_ratio_by_height(img)
    # gray = cv2.imread(image_path, 0)
    # show_pic(gray)
    the_image = gray

    threshed_image = cv2.adaptiveThreshold(the_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    # cv2_helper.Image.show(threshed_image)

    ''' 
        it's really hard.
        It's very depend on the threshed_image.
    '''
    max_contour = find_max_contour(threshed_image)
    square_contour = convert_to_square(max_contour)

    # cv2_helper.Image.show_contours_with_color(the_image, [square_contour])

    larged_contour = cv2_helper.Quadrilateral.enlarge(square_contour, 0.007)
    square_ragion = cv2_helper.Contour.get_rect_ragion(larged_contour, the_image)
    # cv2_helper.Image.show(square_ragion)


    # square_vertices = cv2_helper.Quadrilateral.vertices(square_contour)
    vertical_lines = find_vertical_lines(square_ragion)
    horizontal_lines = find_horizontal_lines(square_ragion)

    # show_lines(square_ragion, vertical_lines+horizontal_lines)

    '''sometimes, the point may less than 0'''
    all_points = tuple(PolarLines.cal_intersection(v_line, h_line) 
            for h_line in horizontal_lines for v_line in vertical_lines)
    def adjust_negative(point):
        x, y = point
        return (max(0,x), max(0,y))
    all_points = map(adjust_negative, all_points)
    all_points = numpy.array(all_points, dtype=numpy.int32)
    # len(all_points).pp()
    # all_points.pp()
    # cv2_helper.Image.show_points_with_color(square_ragion, all_points)

    # v_line = vertical_lines[3]
    # h_line = horizontal_lines[2]
    # v_line.pp()
    # h_line.pp()
    # test_points = [PolarLines.cal_intersection(v_line, h_line)]
    # test_points.pp()
    # cv2_helper.Image.show_points_with_color(square_ragion, test_points)

    # all_points.shape.pp()
    number_contours = cv2_helper.Points.get_quadrilaterals(all_points, SUDOKU_SIZE, SUDOKU_SIZE)
    # number_contours[0:1].pp()
    # cv2_helper.Image.show(the_image)
    # cv2_helper.Image.show_contours_with_color(threshed_square_ragion, number_contours)

    square_ragion = cv2.GaussianBlur(square_ragion,ksize=(5,5), sigmaX=0)
    threshed_square_ragion = cv2.adaptiveThreshold(square_ragion,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=3)
    # threshed_square_ragion = cv2_helper.Image.threshold_white_with_mean_percent(square_ragion)
    # threshed_square_ragion = cv2_helper.Contour.get_rect_ragion(larged_contour, threshed_image)

    get_ragion_func = lambda c: cv2_helper.Contour.get_rect_ragion(c, threshed_square_ragion)
    number_binary_ragions = map(get_ragion_func,  number_contours)

    # cv2_helper.Image.show(gray)
    # cv2_helper.Ragions.show(number_binary_ragions)

    # cv2_helper.Image.show(number_binary_ragions[1])
    # cv2_helper.Image.show(number_binary_ragions[0])
    # for ragion in number_binary_ragions:
    #     ragion.shape.pp()
    # get_nonzero_ragion(number_binary_ragions[0]).pp()

    nonzero_number_rects = map(get_nonzero_rect, number_binary_ragions)
    nonzero_number_rects = [r for r in nonzero_number_rects if r]
    nonzero_number_rects.size().pp()

    nonzero_number_ragions = map(get_nonzero_ragion, number_binary_ragions)

    # test
    all_number_ragion = cv2_helper.Ragions.join_same_size(
        cv2_helper.Ragions.fill_to_same_size(nonzero_number_ragions), 9)
    # cv2_helper.Image.show(threshed_square_ragion)
    # cv2_helper.Image.show(all_number_ragion)
    cv2_helper.Ragions.show([threshed_square_ragion, all_number_ragion])


    # nonzero_number_ragions = tuple(ragion if not numpy_helper.is_array_none(ragion) else numpy.zeros((1,1)) 
    #         for ragion in nonzero_number_ragions)

    # n_1 = get_nonzero_ragion(number_binary_ragions[4])

    # cv2_helper.Image.save_to_txt(numpy_helper.transfer_values_quickly(number_binary_ragions[4],{255:1}),'original_06.txt')
    # cv2_helper.Image.save_to_txt(numpy_helper.transfer_values_quickly(n_1,{255:1}), "number_06.txt")
    # n_1.pp()
    # number_binary_ragions[1].pp()
    # rect_1 = (11, 8, 9, 14)
    # expect_ragion = cv2_helper.Rect.get_ragion(rect_1, number_binary_ragions[1])
    # # cv2_helper.show_rect(number_binary_ragions[1], rect_1)
    # cv2_helper.Image.save_to_txt(numpy_helper.transfer_values_quickly(expect_ragion,{255:1}), "expect_06.txt")
    # nonzero_number_ragions[1].pp()
    # cv2_helper.Image.show(number_ragions[1])
    # cv2_helper.Ragions.show(nonzero_number_ragions)

    # index_ragion_list = [(i, ragion) for i, ragion in index_ragion_list if not is_array_none(ragion)]

def get_nonzero_ragion(the_ragion):
    nonzero_rect = get_nonzero_rect(the_ragion)
    if nonzero_rect:
        return cv2_helper.Rect.get_ragion(nonzero_rect, the_ragion)
    else:
        return numpy.zeros((1,1))


def get_nonzero_rect(the_ragion):
    ''' 
        It will get the rect which has the nonzero in the center and can avoid getting the border.
        But how to choise the center rect is quite important, and sometimes, it will get blank line.
    '''
    ragion_height, ragion_width = the_ragion.shape
    # the_ragion.shape.pp()

    centroid = cv2_helper.Image.centroid(the_ragion)
    # centroid.pp()
    nonzero_indexs = numpy.nonzero(the_ragion)
    nonzero_points = zip(*nonzero_indexs)
    # nonzero_points.pp()

    single_width = int(round(ragion_width*0.18))
    single_height = int(round(ragion_height*0.18))
    # (single_width, single_height).pp()
    original_rect = cv2_helper.Rect.cal_center_rect(centroid, single_width, single_width, single_height, single_height)
    original_rect.p()
    left_x, top_y, edge_width, edge_height = \
            cv2_helper.Rect.cal_center_rect(centroid, single_width, single_width, single_height, single_height)
    # if edge_width > 
    right_x = left_x + edge_width - 1
    bottom_y = top_y + edge_height - 1
    # (left_x, top_y, edge_width, edge_height).pp()
    # top
    top_nonzero = True
    bottom_nonzero = True
    left_nonzero = True
    right_nonzero = True
    while(top_nonzero or bottom_nonzero or left_nonzero or right_nonzero):
        # (left_x, top_y, right_x, bottom_y).pp()
        # (left_nonzero, top_nonzero, right_nonzero, bottom_nonzero).pp()
        # cur_rect = cv2_helper.Rect.create(left_x, top_y, right_x, bottom_y)
        # cv2_helper.Image.show_rect(the_ragion, cur_rect)

        cv2_helper.Image.show
        if top_nonzero:
            top_nonzero = cv2_helper.Rect.has_nonzero((left_x, top_y, right_x-left_x+1, 1),the_ragion)
            if top_nonzero:
                top_y -= 1
            else:
                final_top = top_y + 1
            if top_y <= 0:
                top_nonzero = False
                top_y = 0
                final_top = 0
        if bottom_nonzero:
            bottom_nonzero = cv2_helper.Rect.has_nonzero((left_x, bottom_y, right_x-left_x+1, 1),the_ragion)
            if bottom_nonzero:
                bottom_y += 1
            else:
                final_bottom = bottom_y - 1
            if bottom_y >= ragion_height-1:
                bottom_nonzero = False
                bottom_y = ragion_height-1
                final_bottom = ragion_height-1
        if left_nonzero:
            left_nonzero = cv2_helper.Rect.has_nonzero((left_x, top_y, 1, bottom_y-top_y+1),the_ragion)
            if left_nonzero:
                left_x -= 1
            else:
                final_left = left_x + 1
            if left_x <= 0:
                left_nonzero = False
                left_x = 0
                final_left = 0
        if right_nonzero:
            right_nonzero = cv2_helper.Rect.has_nonzero((right_x, top_y, 1, bottom_y-top_y+1),the_ragion)
            if right_nonzero:
                right_x += 1
            else:
                final_right = right_x - 1 
            if right_x >= ragion_width-1:
                right_nonzero = False
                right_x = ragion_width-1
                final_right = ragion_width-1


    nonzero_rect = cv2_helper.Rect.create(final_left, final_top, final_right, final_bottom)
    # nonzero_rect = (final_left, final_top, final_right-final_left+1, final_bottom-final_top+1)
    # nonzero_rect.pp()
    (final_left, final_top, final_right, final_bottom).pp()
    cv2_helper.Rect.create(final_left-1, final_top-1, final_right+1, final_bottom+1).pp()
    original_rect.pp()
    if nonzero_rect == original_rect:
    # if cv2_helper.Rect.create(final_left-1, final_top-1, final_right+1, final_bottom+1) == original_rect:
        # "NN".p()
        return None
    else:
        return nonzero_rect
        # return cv2_helper.Rect.get_ragion(nonzero_rect, the_ragion)



if __name__ == '__main__':
    from minitest import *
    inject_customized_must_method(numpy.allclose, 'must_close')

    # 1 has one more line,
    # 7 just lost one point, 12 lost many point
    image_path = '../resource/example_pics/sample01.dataset.jpg'
    main(image_path)
    # for i in range(1,15):
    #     pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
    #     pic_file_path.pp()
    #     main(pic_file_path)


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
        # accuracy_pixs.pp()
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
        # all_lines.pp()
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


