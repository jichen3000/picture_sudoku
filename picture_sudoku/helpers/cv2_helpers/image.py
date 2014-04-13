import cv2
import numpy
import operator

from rect import Rect

BLACK = 0
WHITE = 255

class Image(object):
    '''
        notice: in an image, using image[y,x] to get the value of point (x,y).
        The order is reversed with point.
    '''
    @staticmethod
    def centroid(the_image):
        height, width = the_image.shape
        ''' x, y'''
        return (width/2, height/2)

    @staticmethod
    def generate_mask(shape):
        return numpy.zeros(shape, dtype=numpy.uint8)

    @staticmethod
    def fill_contours(the_image, contours, color = WHITE):
        cv2.drawContours(the_image, contours, -1, color, -1)
        return the_image



    @staticmethod
    def resize_keeping_ratio_by_height(the_image, height=700):
        width = float(height) / the_image.shape[0]
        dim = (int(the_image.shape[1] * width), height)
        return cv2.resize(the_image, dim, interpolation = cv2.INTER_AREA)

    @staticmethod
    def cal_nonzero_rect_keeping_ratio(the_image):
        rect = Image.cal_nonzero_rect(the_image)
        return Rect.modify_to_ratio(rect, the_image.shape)

    @staticmethod
    def cal_nonzero_rect(the_image):
        y_indexs, x_indexs = numpy.nonzero(the_image)
        return (min(x_indexs), min(y_indexs), 
                max(x_indexs)-min(x_indexs)+1, max(y_indexs)-min(y_indexs)+1)

    @staticmethod
    def is_not_empty(the_image):
        y_indexs, x_indexs = numpy.nonzero(the_image)
        return (y_indexs.size>0 or x_indexs.size>0)

    @staticmethod
    def save_to_txt(the_image, file_name):
        return numpy.savetxt(file_name, the_image, fmt='%d', delimiter='')


    @staticmethod
    def clip_by_x_y_count(the_image, clip_x_count=2, clip_y_count=2):
        height, width = the_image.shape
        if clip_x_count * 2 > width or clip_y_count * 2 > height:
            raise Exception("out of picture's border")
        return the_image[clip_y_count:height-clip_y_count, 
                clip_x_count:width-clip_x_count]

    @staticmethod
    def clip_by_percent(the_image, percent=0.1):
        height, width = the_image.shape
        clip_y_count = int(round(height * percent))
        clip_x_count = int(round(width * percent))
        return the_image[clip_y_count:height-clip_y_count, 
                clip_x_count:width-clip_x_count]

    @staticmethod
    def clip_by_four_percent(the_image, 
            top_percent=0.1, bottom_percent=0.1, left_percent=0.1, right_percent=0.1):
        height, width = the_image.shape
        clip_top_count = int(round(height * top_percent))
        clip_bottom_count = int(round(height * bottom_percent))
        clip_left_count = int(round(width * left_percent))
        clip_right_count = int(round(width * right_percent))
        return the_image[clip_top_count:height-clip_bottom_count, 
                clip_left_count:width-clip_right_count]

    @staticmethod
    def find_contours(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.01):
        '''
            notice: the threshold_value is the key, if it directly impact the binary matrix. RETR_EXTERNAL
        '''
        contours,not_use = cv2.findContours(threshed_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        def get_approximated_contour(contour):
            perimeter = cv2.arcLength(contour,True)
            # notice the approximation accuracy is the key, if it is 0.01, you will find 4
            # if it is larger than 0.8, you will get nothing at all. 
            approximation_accuracy = accuracy_percent_with_perimeter*perimeter
            return cv2.approxPolyDP(contour,approximation_accuracy,True)

        contours = map(get_approximated_contour, contours)
        if filter_func:
            contours = filter(filter_func, contours)
        return contours

    @staticmethod
    def threshold_white_with_mean_percent(gray_image, mean_percent=0.7):
        threshold_value = int(gray_image.mean()*mean_percent)
        not_use, threshed_image = cv2.threshold(gray_image,threshold_value,WHITE,cv2.THRESH_BINARY_INV)
        return threshed_image


if __name__ == '__main__':
    from minitest import *

    IMG_SIZE = 32

    ORIGINAL_IMAGE_NAME = '../../../resource/example_pics/sample01.dataset.jpg'
    small_number_path = '../../../resource/test/small_number.dataset'
    gray_pic = cv2.imread(ORIGINAL_IMAGE_NAME, 0)
    color_pic = cv2.imread(ORIGINAL_IMAGE_NAME)


    def binary_number_to_lists(file_path):
        with open(file_path) as data_file:
            result = [int(line[index]) for line in data_file 
                for index in range(IMG_SIZE)]
        return result

    def list_to_image_array(the_list, shape=(IMG_SIZE,IMG_SIZE)):
        return numpy.array(the_list, numpy.uint8).reshape(shape)

    with test("Image.centroid"):
        Image.centroid(gray_pic).must_equal((368, 653))



    with test("Image.clip_by_x_y_count"):
        arr = numpy.arange(81).reshape((9,9))
        Image.clip_by_x_y_count(arr).must_equal(
            numpy.array(  [[20, 21, 22, 23, 24],
                           [29, 30, 31, 32, 33],
                           [38, 39, 40, 41, 42],
                           [47, 48, 49, 50, 51],
                           [56, 57, 58, 59, 60]]), numpy.allclose)

    with test("Image.clip_by_percent"):
        arr = numpy.arange(81).reshape((9,9))
        Image.clip_by_percent(arr, 0.2).must_equal(
            numpy.array(  [[20, 21, 22, 23, 24],
                           [29, 30, 31, 32, 33],
                           [38, 39, 40, 41, 42],
                           [47, 48, 49, 50, 51],
                           [56, 57, 58, 59, 60]]), numpy.allclose)

    with test("list_to_image_array"):
        image_list = binary_number_to_lists(small_number_path)
        image_array = list_to_image_array(image_list)
        numpy.count_nonzero(image_array).must_equal(110)
        image_array.shape.must_equal((IMG_SIZE,IMG_SIZE))

    with test("Image.cal_nonzero_rect"):
        image_list = binary_number_to_lists(small_number_path)
        image_array = list_to_image_array(image_list)
        cur_rect = Image.cal_nonzero_rect(image_array)
        cur_rect.must_equal((10, 9, 12, 18))

        ''' uncomment the below, you can see the consequence in a picture. '''
        # cur_contour = rect_to_contour(cur_rect)
        # transfer_values(image_array, {1:255, 0:0})
        # show_contours_in_pic(image_array, [cur_contour], 255)

        ''' uncomment the below, you can see the print consequence. '''
        # sub_image = Rect.get_ragion(cur_rect, image_array)
        # sub_image.pp()


    with test("Image.cal_nonzero_rect_keeping_ratio"):
        image_list = binary_number_to_lists(small_number_path)
        image_array = list_to_image_array(image_list)
        Image.cal_nonzero_rect_keeping_ratio(image_array).must_equal((7, 9, 18, 18))

    with test("Image.is_not_empty"):
        arr = numpy.zeros((3,3))
        Image.is_not_empty(arr).must_equal(False)
        arr[1,1] = 1 
        Image.is_not_empty(arr).must_equal(True)



    with test("Image.fill_contours"):
        mask = Image.generate_mask((600, 400))
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Image.fill_contours(mask, [contour])


