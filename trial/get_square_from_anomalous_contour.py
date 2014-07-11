import numpy
import cv2

from picture_sudoku.helpers.common import Resource, OtherResource
from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.display import Display
from picture_sudoku.cv2_helpers.contour import Contour
from picture_sudoku.cv2_helpers.points import Points
from picture_sudoku.picture_analyzer.polar_lines import PolarLines
from picture_sudoku.picture_analyzer import main_analyzer

SUDOKU_SIZE = 9

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
    Display.polar_lines(square_ragion, vertical_lines+horizontal_lines)
    # Display.polar_lines(square_ragion, horizontal_lines)

    intersections = main_analyzer.find_intersections(vertical_lines, horizontal_lines)
    # intersections.ppl()
    Points.to_contour(intersections).ppl()


def main():
    contour = numpy.load(Resource.get_path("test/max_contour_with_noises.npy"))
    extract_square_from_contour(contour)


if __name__ == '__main__':
    from minitest import *
    main()