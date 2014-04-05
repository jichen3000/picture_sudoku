'''
    this method came from stackoverflow,
    and it cannot work very well for many pictures.
'''

import cv2
import numpy

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



def find_vertical_lines111(gray_pic):
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

    dx = cv2.Sobel(gray_pic,cv2.CV_16S,1,0)
    # dx.pp()
    # dx = cv2.Sobel(gray_pic,cv2.CV_32F,1,0)
    # convert from dtype=int16 to dtype=uint8
    dx = cv2.convertScaleAbs(dx)
    # dx.pp()
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    # cv2_helper.show_pic(dx)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # close = cv2.adaptiveThreshold(dx,255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=2)

    # cv2_helper.show_pic(close)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 8:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()
    # show_pic(closex)
    return closex


def fit_lines(the_image):
    the_image = cv2.GaussianBlur(the_image,ksize=(5,5), sigmaX=0)

    ''' open, it could remove the border '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    the_image = cv2.morphologyEx(the_image, cv2.MORPH_OPEN, kernel)
    # the_image = cv2.erode(the_image, kernel)
    # cv2_helper.Image.show(the_image)
    # the_image = cv2.adaptiveThreshold(the_image,WHITE,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)


    low_threshold, high_threshold = 40, 80

    the_image = cv2.Canny(the_image, low_threshold, high_threshold)

    ''' change the image to uint8'''
    the_image = cv2.convertScaleAbs(the_image)

    contours = cv2_helper.find_contours(the_image, None, 0.01)

    lines = cv2.fitLine(contours[0],distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01, aeps=0.01)

    cv2_helper.Image.show_contours_with_color(the_image, contours[:12])


def find_vertical_lines(gray_pic):
    gray_pic = cv2.GaussianBlur(gray_pic,ksize=(5,5), sigmaX=0)


    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))

    dx = cv2.Sobel(gray_pic,cv2.CV_16S,2,0)
    # dx.pp()
    # dx = cv2.Sobel(gray_pic,cv2.CV_32F,1,0)
    # convert from dtype=int16 to dtype=uint8
    dx = cv2.convertScaleAbs(dx)
    # dx.pp()
    # cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    # dx.pp()
    # cv2_helper.show_pic(dx)
    # ret,close = cv2.threshold(dx,10,255,cv2.THRESH_BINARY)

    low_threshold, high_threshold = 40, 80

    close = cv2.Canny(dx, low_threshold, high_threshold)

    # close.pp()
    cv2_helper.show_pic(close)
    # close = cv2.adaptiveThreshold(dx,255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=2)

    # cv2_helper.show_pic(close)
    # close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
    # cv2_helper.Image.show(close)

    # contours, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     if h/w > 6:
    #         cv2.drawContours(close,[cnt],0,255,-1)
    #     else:
    #         cv2.drawContours(close,[cnt],0,0,-1)
    # def filter_func(contour):
    #     x,y,w,h = cv2.boundingRect(contour)
    #     return h/w > 8
    # # contours = filter(filter_func, contours)
    # # cv2_helper.Image.show_contours_with_color(gray_pic, contours)
    # # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()
    # cv2_helper.Image.show(closex)

    threshold = 120

    lines = cv2.HoughLines(closex, rho=1, theta=numpy.pi/180, threshold= threshold)

    lines = lines[0]
    show_lines(closex, lines)
    return closex

def find_horizontal_lines(gray_pic, square_vertices):

    def filter_line(line):
        # return False
        rho, theta = line
        theta_degree = (theta * 180/ numpy.pi) - 90
        # (theta_degree).p()
        return abs(theta_degree) < 3 + max_angle
        # numpy.tan(theta).pp()
        # return 1 < abs(numpy.tan(theta)) * step
        # (max(top_line_slope, bottom_line_slope) > numpy.tan(theta) > min(top_line_slope, bottom_line_slope)).p()
        # return max(top_line_slope, bottom_line_slope) * 1.3 >= numpy.tan(numpy.pi/2 - theta) \
        #     >= 0.7 * min(top_line_slope, bottom_line_slope)
        # return numpy.tan(theta) * (max(top_line_slope, bottom_line_slope)  >= -1 and 
        #     numpy.tan(theta) * min(top_line_slope, bottom_line_slope)) <= -1

    gray_pic = cv2.GaussianBlur(gray_pic,ksize=(5,5), sigmaX=0)
    # gray_pic = cv2.bilateralFilter(gray_pic, 5, 5, 2)


    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))

    dx = cv2.Sobel(gray_pic,cv2.CV_16S,0,2)
    # dx.pp()
    # dx = cv2.Sobel(gray_pic,cv2.CV_32F,1,0)
    # convert from dtype=int16 to dtype=uint8
    dx = cv2.convertScaleAbs(dx)
    # dx.pp()
    # cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    # dx.pp()
    # cv2_helper.show_pic(dx)
    # ret,close = cv2.threshold(dx,10,255,cv2.THRESH_BINARY)

    low_threshold, high_threshold = 40, 80

    close = cv2.Canny(dx, low_threshold, high_threshold)

    # close.pp()
    # cv2_helper.show_pic(close)
    # close = cv2.adaptiveThreshold(dx,255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=2)

    # cv2_helper.show_pic(close)
    # close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
    # cv2_helper.Image.show(close)

    # contours, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     if h/w > 6:
    #         cv2.drawContours(close,[cnt],0,255,-1)
    #     else:
    #         cv2.drawContours(close,[cnt],0,0,-1)
    # def filter_func(contour):
    #     x,y,w,h = cv2.boundingRect(contour)
    #     return h/w > 8
    # # contours = filter(filter_func, contours)
    # # cv2_helper.Image.show_contours_with_color(gray_pic, contours)
    # # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()
    # cv2_helper.Image.show(closex)

    threshold = 120


    lines = cv2.HoughLines(closex, rho=1, theta=numpy.pi/180, threshold= threshold)
    lines = lines[0]

    while len(lines) > 40 and threshold < 400:
        threshold += 10
        lines = cv2.HoughLines(closex, rho=1, theta=numpy.pi/180, threshold= threshold)
        lines = lines[0]

    (len(lines), threshold).pp()
    square_vertices.pp()

    top_line_slope = cv2_helper.Points.cal_line_slope(square_vertices[0], square_vertices[3])
    top_line_angle = numpy.arctan(top_line_slope) * 180 / numpy.pi
    # square_vertices.p()
    bottom_line_slope = cv2_helper.Points.cal_line_slope(square_vertices[1], square_vertices[2])
    bottom_line_angle = numpy.arctan(bottom_line_slope) * 180 / numpy.pi
    (top_line_angle, bottom_line_angle).pp()
    max_angle = max(abs(top_line_angle), abs(bottom_line_angle))
    # top_line_slope.p()
    # bottom_line_slope.p()
    step = numpy.tan(numpy.pi/18)
    # step.p()

    # lines = lines[:3]
    lines = filter(filter_line, lines)
    show_lines(closex, lines)
    return closex



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


def find_points(gray_pic):
    contours, hier = cv2.findContours(gray_pic,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        mom = cv2.moments(cnt)
        if mom['m00'] == 0:
            continue
        # [mom['m10'],mom['m01'],mom['m00'] ].pp()
        (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
        # cv2.circle(img,(x,y),4,(0,255,0),-1)
        centroids.append((x,y))
    return centroids

def show_points_in_pic(pic_array, points):
    for point in points:
        cv2.circle(pic_array,point,4,(0,255,0),-1)
    show_pic(pic_array)


def main(image_path):
    img = cv2.imread(image_path)
    # img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2_helper.Image.resize_keeping_ratio_by_height(gray)
    img = cv2_helper.Image.resize_keeping_ratio_by_height(img)
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

    larged_contour = cv2_helper.Quadrilateral.enlarge(square_contour, 0.02)
    square_ragion = cv2_helper.Contour.get_rect_ragion(larged_contour, the_image)

    # # fit_lines(square_ragion)
    # vx,vy,x,y = cv2.fitLine(max_contour,distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01, aeps=0.01)
    # # len(line).pp()
    # # line.pp()

    # lefty = int((-x*vy/vx) + y)
    # righty = int(((the_image.shape[1]-x)*vy/vx)+y)

    # cv2.line(the_image,(the_image.shape[1]-1,righty),(0,lefty),255,2)
    # cv2_helper.Image.show(the_image)

    square_vertices = cv2_helper.Quadrilateral.vertices(square_contour)
    # find_vertical_lines(square_ragion)
    find_horizontal_lines(square_ragion, square_vertices)



if __name__ == '__main__':
    from minitest import *
    # image_path = '../resource/example_pics/sample01.dataset.jpg'
    # main(image_path)
    for i in range(1,15):
        pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        pic_file_path.pp()
        main(pic_file_path)

