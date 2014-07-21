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
        height, width = the_image.shape
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
    def resize_keeping_ratio_by_height(the_image, height=700, interpolation = cv2.INTER_AREA):
        width = float(height) / the_image.shape[0]
        dim = (int(the_image.shape[1] * width), height)
        return cv2.resize(the_image, dim, interpolation = interpolation)

    @staticmethod
    def resize_keeping_ratio_by_fixed_length(the_image, fixed_length=32, interpolation = cv2.INTER_AREA):
        ''' (16,18),(16,14),(30,35), (38,36)
        '''
        height, width = the_image.shape
        if height > width:
            sized_height = fixed_length
            ratio = float(fixed_length) / height
            sized_width = int(width * ratio)
        else:
            sized_width = fixed_length
            ratio = float(fixed_length) / width
            sized_height = int(height * ratio)
        return cv2.resize(the_image, (sized_width, sized_height),
                interpolation = interpolation)

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
    def save_to_txt(the_image, file_path):
        '''
            for gray image, actually, if you want use load_from_txt, it should be binary image.
        '''
        return numpy.savetxt(file_path, the_image, fmt='%d', delimiter='')

    @staticmethod
    def load_from_txt(file_path):
        '''
            get image from txt file.
        '''
        line_list_list = []
        with open(file_path) as data_file:
            for line in data_file:
                striped_line = line.strip()
                if len(striped_line):
                    line_list = [int(cur_str) for cur_str in striped_line]
                    line_list_list.append(line_list)
        height = len(line_list_list)
        width = len(line_list_list[0])
        return numpy.array(line_list_list, numpy.uint8).reshape((height, width))



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

    @staticmethod
    def colorize(the_image):
        return cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)

if __name__ == '__main__':
    from minitest import *

    from display import Display
    from picture_sudoku.helpers import numpy_helper

    inject(numpy.allclose, 'must_close')


    ORIGINAL_IMAGE_NAME = '../../resource/example_pics/sample01.dataset.jpg'
    gray_pic = cv2.imread(ORIGINAL_IMAGE_NAME, 0)
    color_pic = cv2.imread(ORIGINAL_IMAGE_NAME)

    small_number_path = '../../resource/test/small_number.dataset'
    small_image = Image.load_from_txt(small_number_path)


    with test(Image.save_to_txt):
        binary_image_path = '../../resource/test/binary_image.npy'
        binary_image = numpy.load(binary_image_path)
        # binary_image.ppl()

        txt_image_path = '../../resource/test/binary_image.dataset'
        Image.save_to_txt(binary_image, txt_image_path)

        loaded_binary_image = Image.load_from_txt(txt_image_path)
        loaded_binary_image.must_close(binary_image)
        # Display.binary_image(loaded_binary_image)
        pass        

    with test(Image.load_from_txt):
        small_image.shape.must_equal((32,32))
        numpy.count_nonzero(small_image).must_equal(110)

    with test(Image.centroid):
        Image.centroid(gray_pic).must_equal((276, 489))

    with test(Image.clip_by_x_y_count):
        arr = numpy.arange(81).reshape((9,9))
        Image.clip_by_x_y_count(arr).must_close(
            numpy.array(  [[20, 21, 22, 23, 24],
                           [29, 30, 31, 32, 33],
                           [38, 39, 40, 41, 42],
                           [47, 48, 49, 50, 51],
                           [56, 57, 58, 59, 60]]) )

    with test(Image.clip_by_percent):
        arr = numpy.arange(81).reshape((9,9))
        Image.clip_by_percent(arr, 0.2).must_close(
            numpy.array(  [[20, 21, 22, 23, 24],
                           [29, 30, 31, 32, 33],
                           [38, 39, 40, 41, 42],
                           [47, 48, 49, 50, 51],
                           [56, 57, 58, 59, 60]]) )


    with test(Image.cal_nonzero_rect):
        cur_rect = Image.cal_nonzero_rect(small_image)
        cur_rect.must_equal((10, 9, 12, 18))

        ''' uncomment the below, you can see the consequence in a picture. '''
        # cur_contour = Rect.to_contour(cur_rect)
        # numpy_helper.transfer_values_quickly(small_image, {1:255, 0:0})
        # Display.contours(small_image, [cur_contour], 255)

        ''' uncomment the below, you can see the print consequence. '''
        # sub_image = Rect.get_ragion(cur_rect, small_image)
        # sub_image.pp()


    with test(Image.cal_nonzero_rect_keeping_ratio):
        Image.cal_nonzero_rect_keeping_ratio(small_image).must_equal((7, 9, 18, 18))

    with test(Image.is_not_empty):
        arr = numpy.zeros((3,3))
        Image.is_not_empty(arr).must_equal(False)
        arr[1,1] = 1 
        Image.is_not_empty(arr).must_equal(True)

    with test(Image.fill_contours):
        mask = Image.generate_mask((600, 400))
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Image.fill_contours(mask, [contour])

    with test(Image.resize_keeping_ratio_by_fixed_length):
        the_shape = (16,18)
        mask = Image.generate_mask(the_shape)
        Image.resize_keeping_ratio_by_fixed_length(mask).shape.must_equal((28, 32))

        the_shape = (16,14)
        mask = Image.generate_mask(the_shape)
        Image.resize_keeping_ratio_by_fixed_length(mask).shape.must_equal((32, 28))

        the_shape = (30,35)
        mask = Image.generate_mask(the_shape)
        Image.resize_keeping_ratio_by_fixed_length(mask).shape.must_equal((27, 32))

        the_shape = (38,36)
        mask = Image.generate_mask(the_shape)
        Image.resize_keeping_ratio_by_fixed_length(mask).shape.must_equal((32, 30))
        


