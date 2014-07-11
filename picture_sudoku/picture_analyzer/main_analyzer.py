import cv2
import numpy
import operator

from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

from picture_sudoku.helpers.exceptions import SudokuError

from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.ragion import Ragion
from picture_sudoku.cv2_helpers.ragion import Ragions
from picture_sudoku.cv2_helpers.rect import Rect
from picture_sudoku.cv2_helpers.contour import Contour
from picture_sudoku.cv2_helpers.quadrilateral import Quadrilateral
from picture_sudoku.cv2_helpers.points import Points
from picture_sudoku.cv2_helpers.display import Display


from polar_lines import PolarLines
import nonzero_rect

# large is brighter, less is darker.
BLACK = 0
WHITE = 255

SUDOKU_SIZE = 9

__all__ = ['extract_number_ragions', 'extract_specified_number_ragion']

def extract_specified_number_ragion(image_path, x, y):
    number_indexs, number_ragions = extract_number_ragions(image_path)
    specified_number_index = (y * SUDOKU_SIZE + x)
    return number_ragions[number_indexs.index(specified_number_index)]

def extract_number_ragions(image_path):
    '''
        It will analyze sudoku image, and return the number ragion with index.
        The result likes:
        ((0, image_ragion_0), (3, image_ragion_3), (8, image_ragion_0) ... )
    '''
    gray_image = cv2.imread(image_path, 0)

    square_ragion = find_square_ragion(gray_image)

    vertical_lines = find_sudoku_vertical_lines(square_ragion)
    if len(vertical_lines) > SUDOKU_SIZE + 1:
        raise SudokuError("The count of vertical_lines is larger than {0}".format(SUDOKU_SIZE))

    horizontal_lines = find_sudoku_horizontal_lines(square_ragion)
    if len(horizontal_lines) > SUDOKU_SIZE + 1:
        raise SudokuError("The count of horizontal_lines is larger than {0}".format(SUDOKU_SIZE))
    # Display.polar_lines(square_ragion, vertical_lines+horizontal_lines)

    intersections = find_intersections(vertical_lines, horizontal_lines)

    cell_ragions = split_cell_ragion(intersections,square_ragion)
    # Display.ragions(cell_ragions)

    index_and_number_ragions = analyze_cell_ragions(cell_ragions)

    number_indexs, number_ragions = zip(*index_and_number_ragions)
    binary_number_ragions = map(lambda x: numpy_helper.transfer_values_quickly(x, {255:1}), number_ragions)
    # Display.ragions([cell_ragions[number_indexs[23]], number_ragions[23]])
    # save_dataset(cell_ragions[number_indexs[5]], 'sample_13_05.dataset')
    # save_dataset(number_ragions[5], 'sample_13_05_nonzeros.dataset')
    # test
    # Display.ragions(cell_ragions)
    # show_all(square_ragion, index_and_number_ragions)
    # show_all(square_ragion, index_and_number_ragions)
    # number_ragions[0].ppl()
    # Display.image(number_ragions[0])

    return number_indexs, binary_number_ragions


def find_square_ragion(gray_image):
    # for some big size pictures which need to blur,
    # but cannot using ksize=(5,5), since some picture will get wrong number value.
    blured_image = cv2.GaussianBlur(gray_image, ksize=(3,3), sigmaX=0)

    threshed_image = cv2.adaptiveThreshold(blured_image,WHITE,
        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    # from picture_sudoku.cv2_helpers.display import Display
    # Display.image(threshed_image)

    ''' 
        It's very depend on the threshed_image.
    '''
    max_contour = find_max_contour(threshed_image)

    # Display.contours(gray_image, [max_contour])
    # max_contour.ppl()
    # 'test'.pl()
    # from picture_sudoku.helpers.common import Resource
    # numpy.save(Resource.get_path("for_issues/max_contour.npy"), max_contour)
    # square_contour = convert_to_square(max_contour)
    square_contour = extract_square_from_contour(max_contour)
    # square_contour.ppl()

    if square_contour == None:
        raise SudokuError("Cannot find sudoku square!")
    # from picture_sudoku.cv2_helpers.display import Display
    # Display.contours(gray_image, [square_contour])

    larged_contour = Quadrilateral.enlarge(square_contour, 0.007)
    larged_contour = Contour.check_beyond_borders(larged_contour, gray_image.shape)
    # from picture_sudoku.cv2_helpers.display import Display
    # Display.contours(gray_image, [larged_contour])
    # larged_contour.ppl()

    square_ragion = Contour.get_rect_ragion(larged_contour, gray_image)
    # Display.image(square_ragion)

    return square_ragion

def find_sudoku_vertical_lines(gray_image):
    gray_image = cv2.GaussianBlur(gray_image,ksize=(5,5), sigmaX=0)

    mean_lines = find_vertical_lines(gray_image)

    all_lines = PolarLines.fill_lost_lines(mean_lines, SUDOKU_SIZE+1)
    # show_lines(cannied_image, all_lines)

    return all_lines


def find_sudoku_horizontal_lines(gray_image):
    gray_image = cv2.GaussianBlur(gray_image,ksize=(5,5), sigmaX=0)

    mean_lines = find_horizontal_lines(gray_image)

    all_lines = PolarLines.fill_lost_lines(mean_lines, SUDOKU_SIZE+1)
    # Display.polar_lines(cannied_image, all_lines)

    return all_lines

def find_vertical_lines(gray_image):
    # gray_image = cv2.GaussianBlur(gray_image,ksize=(5,5), sigmaX=0)
    # let the horizen lines dispear
    dx_image = cv2.Sobel(gray_image,cv2.CV_16S,2,0)
    # convert from dtype=int16 to dtype=uint8
    dx_image = cv2.convertScaleAbs(dx_image)

    # detect the edges
    low_threshold, high_threshold = 40, 80
    cannied_image = cv2.Canny(dx_image, low_threshold, high_threshold)

    lines = PolarLines.find_suitable_lines(cannied_image)
    # show_lines(cannied_image, lines)

    def to_positive_rho(line):
        if line[0] < 0:
            line[0] = abs(line[0])
            line[1] = line[1] - numpy.pi
        return line
    # tansfer negative angle to positive
    lines = map(to_positive_rho, lines)

    def filter_line(line):
        rho, theta = line
        theta_degree = (theta * 180/ numpy.pi)
        return abs(theta_degree) < 10
    # remove the lines which have large angle
    lines = filter(filter_line, lines)
    # show_lines(cannied_image, lines)

    accuracy_pixs = gray_image.shape[1] / SUDOKU_SIZE *0.3 # 9
    catalogued_lines = PolarLines.catalogue_lines(lines, accuracy_pixs)
    mean_lines = PolarLines.cal_mean_lines(catalogued_lines)
    # all_lines = PolarLines.fill_lost_lines(mean_lines, SUDOKU_SIZE+1)
    # show_lines(cannied_image, all_lines)

    return mean_lines



def find_horizontal_lines(gray_image):
    # gray_image = cv2.GaussianBlur(gray_image,ksize=(5,5), sigmaX=0)

    dy_image = cv2.Sobel(gray_image,cv2.CV_16S,0,2)
    # convert from dtype=int16 to dtype=uint8
    dy_image = cv2.convertScaleAbs(dy_image)

    low_threshold, high_threshold = 40, 80
    cannied_image = cv2.Canny(dy_image, low_threshold, high_threshold)

    lines = PolarLines.find_suitable_lines(cannied_image)

    def filter_line(line):
        rho, theta = line
        theta_degree = (theta * 180/ numpy.pi) - 90
        return abs(theta_degree) < 10
    lines = filter(filter_line, lines)
    accuracy_pixs = gray_image.shape[0] / SUDOKU_SIZE *0.3 # 9

    catalogued_lines = PolarLines.catalogue_lines(lines, accuracy_pixs)
    mean_lines = PolarLines.cal_mean_lines(catalogued_lines)
    return mean_lines


def extract_square_from_contour(contour):
    '''
        it will extract square from the contour which could have many noise points.
    '''
    the_image = Image.generate_mask(Contour.get_shape(contour))
    # Display.contours(the_image, [contour])
    cv2.drawContours(the_image, [contour], -1, 255 ,1)
    # Display.image(the_image)
    # lines = PolarLines.find_suitable_lines(the_image)
    square_ragion = the_image
    vertical_lines = find_vertical_lines(square_ragion)
    border_line_count = 2
    if len(vertical_lines) > border_line_count:
        raise SudokuError("The count of vertical border lines is larger than {0}"
            .format(border_line_count))

    horizontal_lines = find_horizontal_lines(square_ragion)
    if len(horizontal_lines) > border_line_count:
        raise SudokuError("The count of horizontal border lines is larger than {0}"
            .format(border_line_count))
    # Display.polar_lines(square_ragion, vertical_lines+horizontal_lines)
    # Display.polar_lines(square_ragion, horizontal_lines)

    intersections = find_intersections(vertical_lines, horizontal_lines)
    # intersections.ppl()
    square_contour =  Points.to_contour(intersections)
    # order the points
    square_contour = Points.to_contour(Quadrilateral.vertices(square_contour))
    # 'test'.pl()
    # Display.contours(the_image,[square_contour])
    return square_contour


def find_intersections(vertical_lines, horizontal_lines):

    '''sometimes, the point may less than 0'''
    intersections = tuple(PolarLines.cal_intersection(v_line, h_line) 
            for h_line in horizontal_lines for v_line in vertical_lines)

    def adjust_negative(point):
        x, y = point
        return (max(0,x), max(0,y))
    intersections = map(adjust_negative, intersections)

    return numpy.array(intersections, dtype=numpy.int32)

def split_cell_ragion(intersections, square_ragion):
    cell_contours = Points.get_quadrilaterals(intersections, SUDOKU_SIZE, SUDOKU_SIZE)

    threshed_square_ragion = square_ragion
    # blured_ragion = cv2.GaussianBlur(square_ragion, ksize=(5,5), sigmaX=0)
    # threshed_square_ragion = cv2.adaptiveThreshold(blured_ragion,WHITE,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=3)
    # test
    # Display.image(threshed_square_ragion)
    get_ragion_func = lambda c: Contour.get_rect_ragion(c, threshed_square_ragion)
    cell_ragions = map(get_ragion_func,  cell_contours)
    def adjust_one(the_ragion):
        # blured_ragion = cv2.GaussianBlur(the_ragion, ksize=(3,3), sigmaX=0)
        # threshed_square_ragion = cv2.adaptiveThreshold(the_ragion,WHITE,
        #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)
        threshed_square_ragion = Image.threshold_white_with_mean_percent(the_ragion, 0.5)
        return threshed_square_ragion
    cell_ragions = map(adjust_one, cell_ragions)
    # 'test'.pl()
    # Display.ragions(cell_ragions)
    return cell_ragions

def analyze_cell_ragions(cell_ragions):
    def gen_cell_tuple(para_tuple):
        index, cell_ragion = para_tuple
        cell_rect = nonzero_rect.analyze_from_center(cell_ragion)
        if cell_rect:
            cell_ragion = Rect.get_ragion(cell_rect, cell_ragion)
            return (index, cell_ragion)
        return False

    all_cell_ragions = map(gen_cell_tuple, enumerate(cell_ragions))
    return filter(lambda x: x, all_cell_ragions)


def find_max_contour(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.0001):
    contours = Image.find_contours(
        threshed_image, filter_func, accuracy_percent_with_perimeter)
    # from picture_sudoku.cv2_helpers.display import Display
    # Display.contours(threshed_image, contours)
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
    # from picture_sudoku.cv2_helpers.display import Display
    # the_image = Image.generate_mask((1000, 1000))
    # Display.contours(the_image, [contour])
    for i in reversed(range(1,10)):
        result = get_approximated_contour(contour, 0.01*i)
        # the_image = Image.generate_mask((1000, 1000))
        # Display.contours(the_image, [result])
        # i.pl()
        # len(result).pl()
        if len(result)==4:
            return result
    return None

def show_all(square_ragion, index_and_number_ragions):
    '''
        just for test
    '''
    all_number_ragion = []
    for index, number_ragion in index_and_number_ragions:
        ragion_index = len(all_number_ragion)
        if ragion_index < index:
            all_number_ragion += [numpy.zeros((1,1)) for i in range(index-ragion_index)]            
        all_number_ragion.append(number_ragion)
    all_number_ragion = Ragions.join_same_size(
        Ragions.fill_to_same_size(all_number_ragion), 9)
    from picture_sudoku.cv2_helpers.display import Display
    Display.ragions([square_ragion, all_number_ragion])


if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display

    inject(numpy.allclose, 'must_close')

    def save_dataset(gray_image, file_name):
        file_path = '../../resource/test/' + file_name
        transfered_image = numpy_helper.transfer_values_quickly(gray_image,{255:1})
        return Image.save_to_txt(transfered_image,file_path)


    with test(extract_number_ragions):
        image_path = '../../resource/example_pics/sample07.dataset.jpg'
        # gray_image = cv2.imread(image_path, 0)
        # Display.image(gray_image)

        number_indexs, number_ragions = extract_number_ragions(image_path)
        number_indexs.must_equal((0, 1, 5, 7, 8, 9, 10, 16, 17, 30, 32, 36, 37, 
            39, 41, 43, 44, 48, 50, 63, 64, 70, 71, 72, 73, 75, 77, 79, 80))
        # number_indexs.must_equal((0, 1, 3, 7, 8, 9, 10, 16, 17, 30, 32, 
        #     36, 37, 39, 41, 43, 44, 48, 50, 63, 64, 70, 71, 72, 73, 75, 77, 79, 80))
        number_ragions.size().must_equal(29)

        # image_path = '../../resource/example_pics/sample07.dataset.jpg'
        # number_indexs, number_ragions = extract_number_ragions(image_path)


        # image_path = '../../resource/example_pics/sample15.dataset.png'
        # number_indexs, number_ragions = extract_number_ragions(image_path)
        # for i in range(1,15):
        #     pic_file_path = '../../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        #     pic_file_path.ppl()
        #     extract_number_ragions(pic_file_path)
        pass

    with test(extract_square_from_contour):
        contour = numpy.load("../../resource/test/max_contour_with_noises.npy")
        extract_square_from_contour(contour).must_close(
            numpy.array(  [[[ 57, 202]],
                           [[ 75, 455]],
                           [[314, 456]],
                           [[332, 204]]]))





