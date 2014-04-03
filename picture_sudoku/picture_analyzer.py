import numpy
import cv2
import os

from picture_sudoku.helpers import cv2_helper
from picture_sudoku.helpers import numpy_helper

# large is brighter, less is darker.
BLACK = cv2_helper.BLACK
WHITE = cv2_helper.WHITE

IMG_SIZE = 32

'''
    notice, the point in the pic_arr, like [478, 128], 
    the first one is the column number, 
    the second one is the row number.
    Don't get them reversely.
'''
def find_sudoku_number_binary_arr(gray_pic_arr):
    '''
        Find all numbers from a picture in which there is a soduku puzzle.
        The number form is a binary numpy.array in which number parts are 1, 
        the others are 0.
    '''
    '''
        notice: the threshold_value is the key, if it directly impact the binary matrix.
    '''
    threshed_pic_array = cv2_helper.threshold_white_with_mean_percent(gray_pic_arr,0.8)
    cv2_helper.show_pic(threshed_pic_array)

    square = find_max_square(threshed_pic_array)
    # cv2_helper.show_contours_in_pic(threshed_pic_array, [square])

    square_rect=cv2.boundingRect(square)
    number_rects = cv2_helper.Rect.cal_split_ragion_rects(square_rect, 9, 9)
    # cv2_helper.show_rects_in_pic(gray_pic_arr, number_rects)


    binary_pic = numpy_helper.transfer_values_quickly(threshed_pic_array, {BLACK:0, WHITE:1})
    number_binary_ragions = map(lambda c: cv2_helper.get_rect_ragion_with_rect(binary_pic, c),
        number_rects)

    number_binary_ragions = map(remove_border, number_binary_ragions)

    non_empty_indexs, number_binary_ragions = get_nonzero_ragions_and_indexs(number_binary_ragions)
    # indexs.pp()

    # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

    number_binary_ragions = map(remove_margin, number_binary_ragions)

    number_binary_ragions = map(enlarge, number_binary_ragions)
    # cv2_helper.show_pic(threshed_pic_array)
    # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

    return number_binary_ragions, non_empty_indexs

def remove_border(pic_array):
    return cv2_helper.clip_array_by_four_percent(pic_array, 
        top_percent=0.06, bottom_percent=0.06, left_percent=0.10, right_percent=0.10)
    # return cv2_helper.clip_array_by_percent(pic_array, percent=0.08)

def is_array_none(contour):
    return numpy.all(contour, None)==None

def is_in_center_area(pic_array, contour, close_percent=0.3):
    mass_x, mass_y = cv2_helper.Contour.mass_center(contour)
    pic_x, pic_y = cv2_helper.Image.center_point(pic_array)
    # (mass_x, mass_y).pp()
    # (pic_x, pic_y).pp()
    result =  ((1-close_percent) * pic_x < mass_x < (1+close_percent) * pic_x) and \
            ((1-close_percent) * pic_y < mass_y < (1+close_percent) * pic_y)
    # result.pp()
    return result


def extract_number(ragion):
    ''' 7: 20, 36'''
    # def find_max_contour222(threshed_pic_array, accuracy_percent_with_perimeter=0.0001):
    #     contours,not_use = cv2.findContours(threshed_pic_array.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #     contours.size().pp()
    #     if len(contours) == 0:
    #         return None
    #     contours = map(get_approximated_contour, contours)
    #     contours = [get_approximated_contour(contour, accuracy_percent_with_perimeter) 
    #             for contour in contours]
    #     # contours = filter(is_almost_contour, contours)
    #     contour_area_arr = [cv2.contourArea(i) for i in contours]
    #     max_contour = contours[contour_area_arr.index(max(contour_area_arr))]
    #     return max_contour

    # max_contour = find_max_contour222(ragion, 0.001)
    # ragion = cv2.adaptiveThreshold(ragion.copy(),WHITE,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=5, C=5)
    # ragion = cv2_helper.threshold_white_with_mean_percent(ragion)
    # cv2_helper.show_pic(ragion)
    def is_in_center_area_func(contour):
        return is_in_center_area(ragion, contour, close_percent=0.5)
    max_contour = find_max_contour(ragion, filter_func = is_in_center_area_func)

    # cv2_helper.show_contours_in_color_pic(ragion, [max_contour])
    if is_array_none(max_contour):
        # 'is_array_none'.pp()
        return None

    # cv2_helper.Contour.mass_center(max_contour).pp()
    # is_in_center_area_func(max_contour).pp()
    # if not is_in_center_area(ragion, max_contour):
    #     return None

    x, y, width, height = cv2.boundingRect(max_contour)
    # cv2_helper.show_rects_in_color_pic(ragion, [(x, y, width, height)])
    # cv2.boundingRect(max_contour).pp()
    ragion_height, ragion_width = ragion.shape
    # if (float(width) / ragion_width) > 0.3 and (float(height) / ragion_height) > 0.5:
    new_ragion = numpy.zeros(ragion.shape, dtype=numpy.uint8)
    new_ragion[y:y+height, x:x+width] = ragion[y:y+height, x:x+width]
    return new_ragion
    return None

def get_nonzero_ragions_and_indexs(ragions):
    '''
        return the ragions which have the nonzero points
    '''
    # index_ragion_list = [(i, ragion) for i, ragion in enumerate(ragions) 
    #         if cv2_helper.is_not_empty_pic(ragion)]

    # cv2_helper.show_same_size_ragions_as_pic(ragions)

    index_ragion_list = [(i, extract_number(ragion)) for i, ragion in enumerate(ragions)]
    index_ragion_list = [(i, ragion) for i, ragion in index_ragion_list if not is_array_none(ragion)]

    ''' todo: 0, 8'''
    ''' need to remove the border without cliping firstly'''
    # for i, ragion in index_ragion_list:
    #     i.pp()
    #     contours = cv2_helper.find_contours(ragion)
    #     len(contours).pp()
    # sp_ragion = index_ragion_list[-1][1]
    # sp_ragion = index_ragion_list[36][1]
    # cv2_helper.show_pic(sp_ragion)
    # sp_ragion = extract_number(sp_ragion)
    # cv2_helper.show_pic(sp_ragion)
    # sp_ragion.pp()
    # new_ragion = numpy.zeros(sp_ragion.shape, dtype=numpy.uint8)
    # sp_ragion.shape.pp()
    # sp_contours = cv2_helper.find_contours(sp_ragion)
    # len(sp_contours).pp()
    # cv2.drawContours(new_ragion, sp_contours, contourIdx=-1, color=WHITE, thickness=2)
    # for sp_contour in sp_contours:
    #     cal_mass_center(sp_contour).pp()
    # cv2_helper.show_pic(new_ragion)
    return zip(*index_ragion_list)



def remove_margin(pic_array):
    new_rect = cv2_helper.cal_nonzero_rect_as_pic_ratio(pic_array)
    return cv2_helper.get_rect_ragion_with_rect(pic_array, new_rect)
    
def enlarge(pic_array):
    return cv2.resize(pic_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR);
    # return cv2.resize(pic_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC);


def find_max_square(threshed_pic_array):
    # squares = threshed_pic_array
    squares = cv2_helper.find_contours(threshed_pic_array, is_almost_square)
    square_perimeter_arr = [cv2.arcLength(i,True) for i in squares]
    return squares[square_perimeter_arr.index(max(square_perimeter_arr))]


def is_almost_square(contour, accuracy=0.001):
    '''
        The accuracy is the key, and cannot larger than 0.001
    '''    
    if len(contour)!=4:
        return False
    perimeter = cv2.arcLength(contour, True)
    area_from_perimeter = (perimeter / 4) ** 2
    real_area = cv2.contourArea(contour)
    if (1-accuracy) * area_from_perimeter < real_area < (1+accuracy) * area_from_perimeter:
        return True
    return False


def analyze_sudoku_pic_to_binary_images(pic_file_path, binary_image_path, suffix = '.dataset'):
    gray_pic_array = cv2.imread(pic_file_path, 0)
    number_binary_arr, sudoku_indexs = find_sudoku_number_binary_arr(gray_pic_array)

    pic_file_name = os.path.basename(pic_file_path).split('.')[0]

    for number_binary, sudoku_index in zip(number_binary_arr, sudoku_indexs):
        file_path = os.path.join(binary_image_path, 
                pic_file_name+"_"+str(sudoku_index).zfill(3)+suffix)
        cv2_helper.save_binary_pic_txt(file_path, number_binary)
        # break

    return True



if __name__ == '__main__':
    from minitest import *

    ORIGINAL_IMAGE_NAME = '../resource/example_pics/sample01.dataset.jpg'
    binary_image_path = '../resource/generated_binaries'

    gray_arr = cv2.imread(ORIGINAL_IMAGE_NAME, 0)
    gray_area_arr = gray_arr[400:1100,50:700]
    color_arr = cv2.imread(ORIGINAL_IMAGE_NAME)
    color_area_arr = color_arr[400:1100,50:700]


    with test("is_almost_square"):
        contour = numpy.array([[[ 671,  421]],
                               [[  78,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]])
        is_almost_square(contour).must_equal(True)
        contour = numpy.array([[[ 671,  421]],
                               [[  128,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]])
        is_almost_square(contour).must_equal(False)


    with test("find_max_square"):
        current_pic_arr = cv2_helper.threshold_white_with_mean_percent(gray_arr)
        max_square = find_max_square(current_pic_arr)
        max_square.must_equal(numpy.array([[[ 671,  421]],
                               [[  78,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]]), numpy.allclose)
        (cv2.arcLength(max_square,True) > 2300).must_equal(True)
        pass



    # with test("show max square in full pic"):
    #     current_pic_arr = color_arr
    #     cv2.drawContours(current_pic_arr,[max_square],-1,(0,255,255),1)
    #     cv2_helper.show_pic(current_pic_arr)

    # with test("show max square in area pic"):
    #     current_pic_arr = cv2_helper.threshold_white_with_mean_percent(gray_area_arr)
    #     # current_pic_arr = color_area_arr
    #     area_max_square = find_max_square(current_pic_arr)
    #     cv2.drawContours(current_pic_arr,[area_max_square],-1,(0,255,255),1)
    #     cv2_helper.show_pic(current_pic_arr)


    # with test("find_sudoku_number_binary_arr"):
    #     number_binary_arr, non_empty_indexs = find_sudoku_number_binary_arr(gray_arr)
    #     non_empty_indexs.size().must_equal(30)
    #     non_empty_indexs[0].must_equal(0)
    #     number_5 = number_binary_arr[0]
    #     black_count = numpy.count_nonzero(number_5)
    #     white_count = numpy.count_nonzero(1-number_5)
    #     row_count, col_count = number_5.shape
    #     (row_count*col_count).must_equal(black_count+white_count)
    #     black_count.must_equal(364)
    #     # number_5 = clip_array_by_fixed_size(number_5,delta_start_y=-5)
    #     # numpy.savetxt("test5.dataset",number_5,fmt="%d", delimiter='')

    def analyze_sudoku_pic_path_to_binary_images(the_path):
        for file_name in os.listdir(the_path):
            file_ext = os.path.splitext(file_name)[-1]
            if file_ext in ['.jpg','.png']:
                os.path.join(the_path, file_name).pp()
                # analyze_sudoku_pic_to_binary_images(
                #         os.path.join(the_path, file_name), binary_image_path)
        return True

    def is_almost(contour, accuracy=0.001):
        '''
            The accuracy is the key, and cannot larger than 0.001
        '''    
        if len(contour)!=4:
            return False
        perimeter = cv2.arcLength(contour, True)
        area_from_perimeter = (perimeter / 4) ** 2
        real_area = cv2.contourArea(contour)
        if (1-accuracy) * area_from_perimeter < real_area < (1+accuracy) * area_from_perimeter:
            return True
        return False

    def remove_vertical_lines(gray_pic):

        dx = cv2.Sobel(gray_pic,cv2.CV_16S,1,0)
        # dx = cv2.Sobel(gray_pic,cv2.CV_32F,1,0)
        dx = cv2.convertScaleAbs(dx)
        cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
        cv2_helper.show_pic(dx)
        # ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # # close = cv2.adaptiveThreshold(dx,255,
        # #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=2)

        close = dx
        kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(1,4))
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
        # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernelx,iterations = 1)
        # close = cv2.morphologyEx(close,cv2.MORPH_GRADIENT,kernelx,iterations = 1)
        cv2_helper.show_pic(close)

        # contours, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours, hier = cv2.findContours(close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # contours = cv2_helper.find_contours(close, None, 0.1)
        len(contours).pp()
        def filter_func(contour):
            x,y,w,h = cv2.boundingRect(contour)
            return h/w > 9
        contours = filter(filter_func, contours)
        len(contours).pp()

        mask = cv2_helper.Image.generate_mask(gray_pic.shape)
        mask = cv2_helper.Image.fill_contours(mask,contours)
        cv2_helper.show_pic(mask)
        # cv2_helper.show_contours_in_color_pic(mask, contours)
        # for cnt in contours:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     if h/w > 5:
        #         cv2.drawContours(close,[cnt],0,255,-1)
        #         cv2.drawContours(gray_pic,[cnt],0,255,-1)

            # else:
            #     cv2.drawContours(close,[cnt],0,0,-1)
        # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
        # closex = close.copy()
        # cv2_helper.show_pic(gray_pic)
        return gray_pic

    def remove_horizontal_lines(gray_pic):
        kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(4,1))
        # dy = gray_pic
        dy = cv2.Sobel(gray_pic,cv2.CV_16S,0,1)
        dy = cv2.convertScaleAbs(dy)
        cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely, iterations = 1)
        # close = cv2.morphologyEx(close,cv2.MORPH_OPEN,kernely, iterations = 1)

        contour, hier = cv2.findContours(close,cv2.MORPH_CLOSE,cv2.CHAIN_APPROX_SIMPLE)
        # contour = cv2_helper.find_contours(close, accuracy_percent_with_perimeter=0.001)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if w/h > 10:
                cv2.drawContours(close,[cnt],0,255,-1)
                cv2.drawContours(gray_pic,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)

        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
        closey = close.copy()
        cv2_helper.show_pic(close)
        return gray_pic
        # return closey


    def show_square(pic_file_path):
        ''' 7, 11 get wrong number, 8 clip too much, 9, 12, 13, 14 get more number,
            7 is so special, put it at last
            8 is almost ok

        '''
        pic_file_path = '../resource/example_pics/sample01.dataset.jpg'
        pic_file_path.pp()
        gray_pic_array = cv2.imread(pic_file_path, 0)
        color_pic_array  = cv2.imread(pic_file_path)
        gray_pic_array = cv2_helper.Image.resize_keeping_ratio_by_height(gray_pic_array)
        color_pic_array = cv2_helper.Image.resize_keeping_ratio_by_height(color_pic_array)
        # threshed_pic_array = cv2.adaptiveThreshold(gray_pic_array,WHITE,
        #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 19, 2)
        threshed_pic_array = cv2.adaptiveThreshold(gray_pic_array,WHITE,
            cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

        max_contour = find_sudoku_sqare(threshed_pic_array)
        max_contour.pp()
        cv2_helper.show_contours_in_pic(color_pic_array, [max_contour])

        def warp_square(pic_array, approximated_square):
            ''' 
                let the approximated_square become a true square.
                
            '''
            square_rect = cv2.boundingRect(approximated_square)
            square_rect = cv2_helper.Rect.adjust_to_minimum(square_rect)
            # square_rect_contour = cv2_helper.rect_to_contour(square_rect)

            # approximated_square.pp()
            # square_rect_contour.pp()

            # approximated_square.shape is (4,1,2)
            approximated_square_float = numpy.float32(approximated_square.copy())
            approximated_square_float = cv2_helper.Quadrilateral.vertices(approximated_square_float)
            # square_rect_contour_float = numpy.float32(square_rect_contour.copy())
            square_rect_contour_float = numpy.float32(cv2_helper.Rect.vertices(square_rect))

            retval = cv2.getPerspectiveTransform(approximated_square_float,square_rect_contour_float)
            dsize = pic_array.shape[::-1]
            warp = cv2.warpPerspective(pic_array,retval,dsize)
            return warp, square_rect

        # four_endpoints = cv2_helper.Quadrilateral.vertices(max_contour)
        # # top_points = cv2_helper.Points.cal_step_points(four_endpoints[:2], 10)
        # internal_points = cv2_helper.Points.cal_internal_points(four_endpoints,10,10)
        # cv2_helper.Image.show_points_with_color(gray_pic_array, internal_points)


        # warp, max_rect = warp_square(gray_pic_array, max_contour)
        # cv2_helper.show_rect(warp, max_rect)
        # cv2_helper.show_pic(warp)
        # cv2_helper.show_pic(gray_pic_array)
        # cv2_helper.show_contours_in_pic(warp, [max_contour, rect_max_contour])

        # sub_warp = gray_pic_array
        sub_warp = cv2_helper.get_rect_ragion_with_contour(threshed_pic_array, max_contour)
        # sub_warp = cv2_helper.get_rect_ragion_with_rect(gray_pic_array, max_rect)
        cv2_helper.show_pic(sub_warp)

        sub_warp = remove_vertical_lines(sub_warp)
        sub_warp = remove_horizontal_lines(sub_warp)
        # cv2_helper.show_pic(sub_warp)


        # # square = rect_max_contour
        # number_rects = cv2_helper.Rect.cal_split_ragion_rects(max_rect, 9, 9)
        # # cv2_helper.show_rects_in_pic(warp, number_rects)

        # # threshed_pic_array = warp
        # # binary_pic = warp
        # # binary_pic = numpy_helper.transfer_values_quickly(binary_pic, {BLACK:0, WHITE:1})
        # number_binary_ragions = map(lambda c: cv2_helper.get_rect_ragion_with_rect(warp, c),
        #     number_rects)
            
        # # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

        # number_binary_ragions = map(cv2_helper.threshold_white_with_mean_percent, number_binary_ragions)
        # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)
        # # number_binary_ragions[0].pp()
        # # cv2_helper.show_pic(number_binary_ragions[1])

        # number_binary_ragions = map(remove_border, number_binary_ragions)
        # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

        # non_empty_indexs, number_binary_ragions = get_nonzero_ragions_and_indexs(number_binary_ragions)
        # # non_empty_indexs.pp()
        # non_empty_indexs.size().p()
        # non_empty_indexs.p()

        # # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

        # number_binary_ragions = map(enlarge, number_binary_ragions)
        # # cv2_helper.show_pic(threshed_pic_array)
        # cv2_helper.show_same_size_ragions_as_pic(number_binary_ragions, 9)

    def get_approximated_contour(contour, accuracy_percent_with_perimeter=0.0001):
        perimeter = cv2.arcLength(contour,True)
        # notice the approximation accuracy is the key, if it is 0.01, you will find 4
        # if it is larger than 0.8, you will get nothing at all.
        approximation_accuracy = accuracy_percent_with_perimeter*perimeter
        return cv2.approxPolyDP(contour,approximation_accuracy,True)


    def find_max_contour(threshed_pic_array, filter_func = None, accuracy_percent_with_perimeter=0.0001):
        contours = cv2_helper.find_contours(
            threshed_pic_array, filter_func, accuracy_percent_with_perimeter)
        if len(contours) == 0:
            return None
        contour_area_arr = [cv2.contourArea(i) for i in contours]
        max_contour = contours[contour_area_arr.index(max(contour_area_arr))]
        return max_contour

    def find_sudoku_sqare(threshed_pic_array):
        ''' 
            it's really hard.
            It's very depend on the threshed_pic_array.
        '''
        max_contour = find_max_contour(threshed_pic_array)
        for i in reversed(range(1,10)):
            result = get_approximated_contour(max_contour, 0.01*i)
            # i.p()
            # len(result).p()
            if len(result)==4:
                return result
        return None


    def test_find_sudoku_sqare():
        pic_file_path = '../resource/example_pics/sample14.dataset.jpg'
        gray_pic_array = cv2.imread(pic_file_path, 0)
        color_pic_array  = cv2.imread(pic_file_path)
        gray_pic_array = cv2_helper.resize_with_fixed_height(gray_pic_array)
        color_pic_array = cv2_helper.resize_with_fixed_height(color_pic_array)
        # threshed_pic_array = cv2.adaptiveThreshold(gray_pic_array,WHITE,
        #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 19, 2)
        ''' I've tried 3,3 19,2, 5,5 9,2'''
        threshed_pic_array = cv2.adaptiveThreshold(gray_pic_array,WHITE,
            cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=5, C=2)
        # cv2_helper.show_pic(color_pic_array)
        cv2_helper.show_pic(threshed_pic_array)

        max_contour = find_sudoku_sqare(threshed_pic_array)
        cv2_helper.show_contours_in_pic(color_pic_array, [max_contour])


    # with test('minAreaRect'):
    #     min_max_rect = cv2.minAreaRect(max_contour)
    #     min_max_rect.pp()
    #     points = cv2.cv.BoxPoints(min_max_rect) 
    #     points = numpy.int0(numpy.around(points))
    #     cv2.polylines(color_pic_array,[points],True,(255,0,0),2)
    #     cv2_helper.show_pic(color_pic_array)
 
    with test("main"):
        pic_file_path = '../resource/example_pics/sample01.dataset.jpg'
        gray_pic_array = cv2.imread(pic_file_path, 0)
        color_pic_array  = cv2.imread(pic_file_path)
        gray_pic_array = cv2_helper.Image.resize_keeping_ratio_by_height(gray_pic_array)
        color_pic_array = cv2_helper.Image.resize_keeping_ratio_by_height(color_pic_array)

        # find_sudoku_number_binary_arr2(gray_pic_array)
        # for i in range(1,15):
        #     pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        #     show_square(pic_file_path)
        show_square(pic_file_path)
        # analyze_sudoku_pic_to_binary_images(ORIGINAL_IMAGE_NAME, binary_image_path)
        # analyze_sudoku_pic_to_binary_images('../resource/example_pics/sample02.dataset.jpg', binary_image_path)
        # analyze_sudoku_pic_path_to_binary_images('../resource/example_pics/')

        # test_find_sudoku_sqare()
        pass

