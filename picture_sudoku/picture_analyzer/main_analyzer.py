import cv2
import numpy
import operator

from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

from picture_sudoku.helpers.cv2_helpers.image import Image
from picture_sudoku.helpers.cv2_helpers.ragion import Ragion
from picture_sudoku.helpers.cv2_helpers.ragion import Ragions
from picture_sudoku.helpers.cv2_helpers.rect import Rect
from picture_sudoku.helpers.cv2_helpers.contour import Contour
from picture_sudoku.helpers.cv2_helpers.quadrilateral import Quadrilateral
from picture_sudoku.helpers.cv2_helpers.points import Points


from polar_lines import PolarLines
import cores

# large is brighter, less is darker.
BLACK = 0
WHITE = 255

IMG_SIZE = 32
SUDOKU_SIZE = 9
SMALL_COUNT = 3

def find_max_contour(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.0001):
    contours = Image.find_contours(
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
        # i.pl()
        # len(result).pl()
        if len(result)==4:
            return result
    return None


def cal_horizontal_max_angle(square_vertices):
    ''' it could be used in the filter lines'''
    square_vertices.ppl()

    top_line_slope = Points.cal_line_slope(square_vertices[0], square_vertices[3])
    top_line_angle = numpy.arctan(top_line_slope) * 180 / numpy.pi
    # square_vertices.pl()
    bottom_line_slope = Points.cal_line_slope(square_vertices[1], square_vertices[2])
    bottom_line_angle = numpy.arctan(bottom_line_slope) * 180 / numpy.pi
    (top_line_angle, bottom_line_angle).ppl()
    max_angle = max(abs(top_line_angle), abs(bottom_line_angle))
    max_angle.pl()
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
    # lines.ppl()

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
    # lines.ppl()

    # lines.ppl()
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
    # lines.ppl()

    # show_lines(close, lines)

    accuracy_pixs = gray_pic.shape[0] / SUDOKU_SIZE *0.3 # 9
    line_count = SUDOKU_SIZE+1
    # all_lines = PolarLines.cal_all_lines(lines, accuracy_pixs, SUDOKU_SIZE+1)
    catalogued_lines = PolarLines.catalogue_lines(lines, accuracy_pixs)
    # accuracy_pixs.pl()
    # catalogued_lines.ppl()
    mean_lines = PolarLines.cal_mean_lines(catalogued_lines)
    all_lines = PolarLines.fill_lost_lines(mean_lines, line_count)

    # show_lines(close, all_lines)



    return all_lines

def main(image_path):
    img = cv2.imread(image_path)
    # img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = Image.resize_keeping_ratio_by_height(gray)
    # img = Image.resize_keeping_ratio_by_height(img)
    # gray = cv2.imread(image_path, 0)

    # 'gray show'.pl()
    # Display.image(gray)
    the_image = gray

    threshed_image = cv2.adaptiveThreshold(the_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    # Display.image(threshed_image)

    ''' 
        it's really hard.
        It's very depend on the threshed_image.
    '''
    max_contour = find_max_contour(threshed_image)
    square_contour = convert_to_square(max_contour)

    # Display.contours(the_image, [square_contour])

    larged_contour = Quadrilateral.enlarge(square_contour, 0.007)
    square_ragion = Contour.get_rect_ragion(larged_contour, the_image)
    # Display.image(square_ragion)


    # square_vertices = Quadrilateral.vertices(square_contour)
    vertical_lines = find_vertical_lines(square_ragion)
    horizontal_lines = find_horizontal_lines(square_ragion)

    Display.polar_lines(square_ragion, vertical_lines+horizontal_lines)

    '''sometimes, the point may less than 0'''
    all_points = tuple(PolarLines.cal_intersection(v_line, h_line) 
            for h_line in horizontal_lines for v_line in vertical_lines)
    def adjust_negative(point):
        x, y = point
        return (max(0,x), max(0,y))
    all_points = map(adjust_negative, all_points)
    all_points = numpy.array(all_points, dtype=numpy.int32)
    # len(all_points).ppl()
    # all_points.ppl()
    Display.points(square_ragion, all_points)

    # v_line = vertical_lines[3]
    # h_line = horizontal_lines[2]
    # v_line.ppl()
    # h_line.ppl()
    # test_points = [PolarLines.cal_intersection(v_line, h_line)]
    # test_points.ppl()
    # Display.points(square_ragion, test_points)

    # all_points.shape.ppl()
    number_contours = Points.get_quadrilaterals(all_points, SUDOKU_SIZE, SUDOKU_SIZE)
    # number_contours[0:1].ppl()
    # Display.image(the_image)
    # Display.contours(threshed_square_ragion, number_contours)

    square_ragion = cv2.GaussianBlur(square_ragion,ksize=(5,5), sigmaX=0)
    threshed_square_ragion = cv2.adaptiveThreshold(square_ragion,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=3)
    # threshed_square_ragion = Image.threshold_white_with_mean_percent(square_ragion)
    # threshed_square_ragion = Contour.get_rect_ragion(larged_contour, threshed_image)

    get_ragion_func = lambda c: Contour.get_rect_ragion(c, threshed_square_ragion)
    number_binary_ragions = map(get_ragion_func,  number_contours)


    # Display.image(number_binary_ragions[1])
    # Display.image(number_binary_ragions[0])
    # for ragion in number_binary_ragions:
    #     ragion.shape.ppl()
    # get_nonzero_ragion(number_binary_ragions[0]).ppl()

    nonzero_number_rects = map(cores.get_nonzero_rect, number_binary_ragions)
    nonzero_number_rects = [r for r in nonzero_number_rects if r]
    # nonzero_number_rects.size().ppl()

    nonzero_number_ragions = map(get_nonzero_ragion, number_binary_ragions)

    # test
    all_number_ragion = Ragions.join_same_size(
        Ragions.fill_to_same_size(nonzero_number_ragions), 9)


    # Display.image(threshed_square_ragion)
    # Display.image(all_number_ragion)
    # 'show([threshed_square_ragion, all_number_ragion])'.pl()
    Display.ragions([threshed_square_ragion, all_number_ragion])


    # nonzero_number_ragions = tuple(ragion if not numpy_helper.is_array_none(ragion) else numpy.zeros((1,1)) 
    #         for ragion in nonzero_number_ragions)

    # n_1 = get_nonzero_ragion(number_binary_ragions[4])

    # Image.save_to_txt(numpy_helper.transfer_values_quickly(number_binary_ragions[4],{255:1}),'original_06.txt')
    # Image.save_to_txt(numpy_helper.transfer_values_quickly(n_1,{255:1}), "number_06.txt")
    # n_1.ppl()
    # number_binary_ragions[1].ppl()
    # rect_1 = (11, 8, 9, 14)
    # expect_ragion = Rect.get_ragion(rect_1, number_binary_ragions[1])
    # # show_rect(number_binary_ragions[1], rect_1)
    # Image.save_to_txt(numpy_helper.transfer_values_quickly(expect_ragion,{255:1}), "expect_06.txt")
    # nonzero_number_ragions[1].ppl()
    # Display.image(number_ragions[1])
    # Ragions.show(nonzero_number_ragions)

    # index_ragion_list = [(i, ragion) for i, ragion in index_ragion_list if not is_array_none(ragion)]

def get_nonzero_ragion(the_ragion):
    nonzero_rect = cores.get_nonzero_rect(the_ragion)
    if nonzero_rect:
        return Rect.get_ragion(nonzero_rect, the_ragion)
    else:
        return numpy.zeros((1,1))





if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.helpers.cv2_helpers.display import Display

    inject_customized_must_method(numpy.allclose, 'must_close')

    image_path = '../../resource/example_pics/sample01.dataset.jpg'
    main(image_path)
    # for i in range(1,15):
    #     pic_file_path = '../../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
    #     pic_file_path.ppl()
    #     main(pic_file_path)


