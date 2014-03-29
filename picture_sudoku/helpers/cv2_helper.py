'''
    Test in the codes/python/projects/font_number_binary/cv2_helper.py
'''


import cv2
import numpy



BLACK = 0
WHITE = 255


def get_rect_ragion_with_contour(pic_array, contour):
    # x, y, width, height = cv2.boundingRect(contour)
    # return pic_array[y:y+height,x:x+width]
    return get_rect_ragion_with_rect(pic_array, cv2.boundingRect(contour))

def get_rect_ragion_with_rect(pic_array, rect):
    x, y, width, height = rect
    return pic_array[y:y+height,x:x+width]

def rect_to_contour(rect):
    '''
        The contour could be different with the origanal one 
        which generate the rect by cv2.boundingRect(contour),
        since the rect cannot represent all information.
        And the other difference would be normally the contour.shape 
        is (4,1,2), but this one is (4,2)
    '''
    x, y, width, height = rect
    return numpy.array([(x,         y),
                        (x,         y+height-1),
                        (x+width-1, y+height-1),
                        (x+width-1, y)])

def cal_nonzero_rect(pic_array):
    y_indexs, x_indexs = numpy.nonzero(pic_array)
    return (min(x_indexs), min(y_indexs), 
            max(x_indexs)-min(x_indexs)+1, max(y_indexs)-min(y_indexs)+1)

def is_not_empty_pic(pic_array):
    y_indexs, x_indexs = numpy.nonzero(pic_array)
    return (y_indexs.size>0 or x_indexs.size>0)


def modify_rect_as_pic_ratio(rect, shape):
    '''
        This method will return the rect which has the same ratio 
        of row count and col count
    '''
    x, y, width, height = rect
    pic_height, pic_width= shape
    #height need to be enlarge
    if pic_height * width > pic_width * height:
        new_height = int((pic_height * width) / pic_width)
        delta = int( (new_height - height)/2 )
        new_y = y - delta
        if new_y < 0:
            new_y = 0
        return x, new_y, width, new_height
    elif pic_height * width < pic_width * height:
        new_width = int((pic_width * height) / pic_height)
        delta = int((new_width-width) / 2)
        new_x = x - delta
        if new_x < 0:
            new_x = 0
        return new_x, y, new_width, height
    return rect

def cal_nonzero_rect_as_pic_ratio(pic_array):
    rect = cal_nonzero_rect(pic_array)
    return modify_rect_as_pic_ratio(rect, pic_array.shape)

def get_contour_points(contour):
    ''' another way to get the rect of a contour
        just like boundingRect
    '''
    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect)
    box = numpy.int0(box)
    return box


def cal_split_ragion_rects(contour, split_x_num, split_y_num):
    '''
        Calculate the split ragion rect in a contour.
        The rect is a tuple, like (x, y, width, height)
    '''
    # x, y, width, height = cv2.boundingRect(contour)
    # x_step = int(width / split_x_num)
    # y_step = int(height / split_y_num)
    # result = tuple((x+i*x_step, y+j*y_step, x_step, y_step) 
    #     for j in range(split_y_num) for i in range(split_x_num))
    return Rect.cal_split_ragion_rects(cv2.boundingRect(contour), split_x_num, split_y_num)



def save_binary_pic_txt(file_name, binary_pic):
    return numpy.savetxt(file_name, binary_pic, fmt='%d', delimiter='')

# def clip_array_by_fixed_size(pic_array, fixed_height=32, fixed_width=32, delta_start_y=3):
#     height, width = pic_array.shape
#     if fixed_height > height or fixed_width > width:
#         raise Exception("out of picture's border %d > %d or %d > %d" %(fixed_height, height, fixed_width, width))
#     start_y = int((height - fixed_height)/2)-delta_start_y
#     if start_y < 0:
#         start_y = 0
#     start_x = int((width - fixed_width)/2)
#     return pic_array[start_y:start_y+fixed_height, start_x:start_x+fixed_width]

def clip_array_by_x_y_count(pic_array, clip_x_count=2, clip_y_count=2):
    height, width = pic_array.shape
    if clip_x_count * 2 > width or clip_y_count * 2 > height:
        raise Exception("out of picture's border")
    return pic_array[clip_y_count:height-clip_y_count, 
            clip_x_count:width-clip_x_count]

def clip_array_by_percent(pic_array, percent=0.1):
    height, width = pic_array.shape
    clip_y_count = int(round(height * percent))
    clip_x_count = int(round(width * percent))
    return pic_array[clip_y_count:height-clip_y_count, 
            clip_x_count:width-clip_x_count]

def clip_array_by_four_percent(pic_array, 
        top_percent=0.1, bottom_percent=0.1, left_percent=0.1, right_percent=0.1):
    height, width = pic_array.shape
    clip_top_count = int(round(height * top_percent))
    clip_bottom_count = int(round(height * bottom_percent))
    clip_left_count = int(round(width * left_percent))
    clip_right_count = int(round(width * right_percent))
    return pic_array[clip_top_count:height-clip_bottom_count, 
            clip_left_count:width-clip_right_count]


def find_contours(threshed_pic_array, filter_func = None, accuracy_percent_with_perimeter=0.01):
    '''
        notice: the threshold_value is the key, if it directly impact the binary matrix.
    '''
    contours,not_use = cv2.findContours(threshed_pic_array.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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

def threshold_white_with_mean_percent(gray_pic, mean_percent=0.7):
    threshold_value = int(gray_pic.mean()*mean_percent)
    not_use, threshed_pic_array = cv2.threshold(gray_pic,threshold_value,WHITE,cv2.THRESH_BINARY_INV)
    return threshed_pic_array

def resize_with_fixed_height(pic_array, height=700):
    width = float(height) / pic_array.shape[0]
    dim = (int(pic_array.shape[1] * width), height)
    return cv2.resize(pic_array, dim, interpolation = cv2.INTER_AREA)


class Rect(object):
    '''
        x, y, width, height = rect
    '''
    @staticmethod
    def cal_split_ragion_rects(rect, split_x_num, split_y_num):
        x, y, width, height = rect
        x_step = int(width / split_x_num)
        y_step = int(height / split_y_num)
        result = tuple((x+i*x_step, y+j*y_step, x_step, y_step) 
            for j in range(split_y_num) for i in range(split_x_num))
        return result

    @staticmethod
    def adjust_to_minimum(rect):
        x, y, width, height = rect
        min_width_or_height = min(width, height)

        return (x, y, min_width_or_height, min_width_or_height)

class Contour(object):
    @staticmethod
    def mass_center(contour):
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return (1, 1)
        return ( moments['m10']/moments['m00'], moments['m01']/moments['m00'])

    @staticmethod
    def four_endpoints(contour):
        '''
            It will adjust points to top-left, top-right, bottom-right, bottom-left.
            So it asks the contour is a rectangle
        '''
        points = contour.copy().reshape((4,2))
        results = points.copy()
        x_mean, y_mean = numpy.mean(points, axis=0)
        for point in points:
            if point[0] < x_mean and point[1] < y_mean:
                results[0] = point
            if point[0] < x_mean and point[1] > y_mean:
                results[1] = point
            if point[0] > x_mean and point[1] > y_mean:
                results[2] = point
            if point[0] > x_mean and point[1] < y_mean:
                results[3] = point
        return results



class Points(object):
    @staticmethod
    def cal_step_points(two_endpoints_in_line, step_count):
        '''
            two_endpoints_in_line are the two point of a line.
            step_count will be the points on that line between the two endpoints.
        '''
        step_distance = numpy.float32(two_endpoints_in_line[-1] - two_endpoints_in_line[0]) / step_count
        points_list = [(two_endpoints_in_line[0] + index * step_distance)
                for index in range(step_count+1)]
        return numpy.array(numpy.int32(points_list)).reshape((10,2))
    
    @staticmethod
    def cal_internal_points(four_endpoints_in_rectangle, x_step_count, y_step_count):
        left_points = Points.cal_step_points(four_endpoints_in_rectangle[0:2], y_step_count)
        rifht_points = Points.cal_step_points(four_endpoints_in_rectangle[::-1][0:2], y_step_count)
        internal_points = [] 
        for left_point, right_point in zip(left_points, rifht_points):
            # (left_point, right_point).pp()
            endpoints = numpy.append(left_point, right_point).reshape(2,2)
            internal_points.append(Points.cal_step_points(endpoints, x_step_count))

        # for index in range(x_step_count+1):
        # numpy.append(points[0], points[1]).reshape(2,2)
        return numpy.array(internal_points).reshape((x_step_count+1)*(y_step_count+1),2)

'''
    The below methods are for test.
'''

class Image(object):
    @staticmethod
    def center_point(the_image):
        height, width = the_image.shape
        ''' x, y'''
        return (width/2, height/2)

    @staticmethod
    def show_points_with_color(the_image, points):
        color_image = cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)
        for point in points:
            point = tuple(point)
            point.pp()
            cv2.circle(color_image,point,4,(0,255,0),-1)
        show_pic(color_image)

    @staticmethod
    def generate_mask(shape):
        return numpy.zeros(shape, dtype=numpy.uint8)

    @staticmethod
    def fill_contours(the_image, contours, color = WHITE):
        cv2.drawContours(the_image, contours, -1, color, -1)
        return the_image

    @staticmethod
    def show(the_image, image_name='image'):
        cv2.imshow(image_name, the_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_pic(the_image):
    Image.show(the_image)
    # # cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
    # # cv2.setWindowProperty("pic", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    # # cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
    # # cv2.namedWindow('pic', cv2.WINDOW_AUTOSIZE | cv2.cv.CV_GUI_EXPANDED)
    # cv2.imshow('pic', pic_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def show_contours_in_pic(pic_array, contours, color=(0,255,255)):
    cv2.drawContours(pic_array,contours, -1, color ,1)
    show_pic(pic_array)

def show_contours_in_color_pic(pic_array, contours, color=(0,255,255)):
    show_contours_in_pic(cv2.cvtColor(pic_array, cv2.COLOR_GRAY2BGR), contours, color)

def show_rects_in_pic(pic_array, rects, color=(0,255,255)):
    contours = map(rect_to_contour, rects)
    show_contours_in_pic(pic_array, contours, color)

def show_rects_in_color_pic(pic_array, rects, color=(0,255,255)):
    show_rects_in_pic(cv2.cvtColor(pic_array, cv2.COLOR_GRAY2BGR), rects, color)

def show_rect(pic_array, rect):
    sub_pic_array = get_rect_ragion_with_rect(pic_array, rect)
    show_pic(sub_pic_array)

def show_points_in_color_pic(pic_array, points):
    for point in points:
        cv2.circle(cv2.cvtColor(pic_array, cv2.COLOR_GRAY2BGR),point,4,(0,255,0),-1)
    show_pic(pic_array)



def show_same_size_ragions_as_pic(ragions, count_in_row=9, init_value=BLACK):
    pic_array = join_ragions_as_pic(ragions, count_in_row, init_value)
    show_pic(pic_array)

def join_ragions_as_pic(ragions, count_in_row, init_value=BLACK):
    '''
        The ragion must have same size.
        for test
    '''
    ragion_count = len(ragions)
    # ragion_count.pp()
    ragion_row_count = int(ragion_count / count_in_row) + 1
    steps = 4
    ragion_height, ragion_width = ragions[0].shape
    # ragions[0].shape.pp()
    width = count_in_row * ragion_width + (count_in_row + 1) * steps
    height = ragion_row_count * ragion_height + (ragion_row_count + 1) * steps

    pic_array = numpy.zeros((height, width), dtype=numpy.uint8) + init_value
    # pic_array.shape.pp()
    # show_pic(pic_array)

    for i in range(ragion_row_count):
        for j in range(count_in_row):
            ragion_index = j+i * count_in_row
            if ragion_index >= ragion_count:
                break
            x_index = (i+1)*steps+i*ragion_height
            y_index = (j+1)*steps+j*ragion_width
            pic_array[x_index:x_index+ragion_height, y_index:y_index+ragion_width] = ragions[ragion_index]
            # if ragion_index == 1:
            #     aa = pic_array[x_index:x_index+ragion_height, y_index:y_index+ragion_width] 
            #     aa.shape.pp()
            #     ragions[ragion_index].shape.pp()
            #     aa.pp()
            #     ragions[ragion_index].pp()
            #     # x_index.pp()
            #     # y_index.pp()
            #     show_pic(aa)
    return pic_array


if __name__ == '__main__':
    from minitest import *

    IMG_SIZE = 32

    ORIGINAL_IMAGE_NAME = './images/antiqua1.png'
    gray_pic = cv2.imread(ORIGINAL_IMAGE_NAME, 0)
    color_pic = cv2.imread(ORIGINAL_IMAGE_NAME)

    small_number_path = 'test_resources/small_number.dataset'


    def binary_number_to_lists(file_path):
        with open(file_path) as data_file:
            result = [int(line[index]) for line in data_file 
                for index in range(IMG_SIZE)]
        return result

    def list_to_image_array(the_list, shape=(IMG_SIZE,IMG_SIZE)):
        return numpy.array(the_list, numpy.uint8).reshape(shape)

    with test("Image.center_point"):
        Image.center_point(gray_pic).must_equal((247, 117))

    with test("get_rect_ragion"):
        contour = numpy.array([[[ 171,  21]],
                               [[  18,  26]],
                               [[  25, 216]],
                               [[ 175, 212]]])
        '''
            notice the result will be different with different picture.
        '''
        cur_area_pic1 = get_rect_ragion_with_contour(gray_pic, contour)
        cur_area_pic1.shape.must_equal((196, 158))
        cur_area_pic1[0,0].must_equal(gray_pic[18,21])

        rect = (18, 21, 158, 196)
        cv2.boundingRect(contour).must_equal(rect)
        cur_area_pic2 = get_rect_ragion_with_rect(gray_pic,rect)
        cur_area_pic2.shape.must_equal((196, 158))
        cur_area_pic2[0,0].must_equal(gray_pic[18,21])
        ''' uncomment the below, you can see the consequence in a picture. '''
        # show_pic(cur_area_pic1)    

    with test("cal_split_ragion_rects"):
        contour = numpy.array(
              [[[ 16, 139]],
               [[ 16, 225]],
               [[478, 225]],
               [[478, 139]]])
        ragion_rects = cal_split_ragion_rects(contour, 10, 1)
        ragion_contours = map(rect_to_contour, ragion_rects)
        ragion_contours.size().must_equal(10)
        ''' uncomment the below, you can see the consequence in a picture. '''        
        # show_contours_in_pic(color_pic, ragion_contours)
        pass

    with test("rect_to_contour"):
        contour = numpy.array([[[ 171,  21]],
                               [[  18,  26]],
                               [[  25, 216]],
                               [[ 175, 212]]])
        rect = (18, 21, 158, 196)
        rect_to_contour(rect).must_equal(
            numpy.array([  [ 18,  21],
                           [ 18, 216],
                           [175, 216],
                           [175,  21]]), numpy.allclose)

    with test("clip_array_by_x_y_count"):
        arr = numpy.arange(81).reshape((9,9))
        clip_array_by_x_y_count(arr).must_equal(
            numpy.array(  [[20, 21, 22, 23, 24],
                           [29, 30, 31, 32, 33],
                           [38, 39, 40, 41, 42],
                           [47, 48, 49, 50, 51],
                           [56, 57, 58, 59, 60]]), numpy.allclose)

    with test("clip_array_by_percent"):
        arr = numpy.arange(81).reshape((9,9))
        clip_array_by_percent(arr, 0.2).must_equal(
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

    with test("cal_nonzero_rect"):
        image_list = binary_number_to_lists(small_number_path)
        image_array = list_to_image_array(image_list)
        cur_rect = cal_nonzero_rect(image_array)
        cur_rect.must_equal((10, 9, 12, 18))

        ''' uncomment the below, you can see the consequence in a picture. '''
        # cur_contour = rect_to_contour(cur_rect)
        # transfer_values(image_array, {1:255, 0:0})
        # show_contours_in_pic(image_array, [cur_contour], 255)

        ''' uncomment the below, you can see the print consequence. '''
        # sub_image = get_rect_ragion_with_rect(image_array, cur_rect)
        # sub_image.pp()

    with test("modify_rect_as_pic_ratio"):
        cur_rect = (10, 9, 12, 18)
        modify_rect_as_pic_ratio(cur_rect, (IMG_SIZE, IMG_SIZE)).must_equal(
            (7, 9, 18, 18))

    with test("cal_nonzero_rect_as_pic_ratio"):
        image_list = binary_number_to_lists(small_number_path)
        image_array = list_to_image_array(image_list)
        cal_nonzero_rect_as_pic_ratio(image_array).must_equal((7, 9, 18, 18))

    with test("is_not_empty_pic"):
        arr = numpy.zeros((3,3))
        is_not_empty_pic(arr).must_equal(False)
        arr[1,1] = 1 
        is_not_empty_pic(arr).must_equal(True)

    with test("Rect.adjust_to_minimum"):
        rect = (153, 13, 668, 628)
        Rect.adjust_to_minimum(rect).must_equal((153, 13, 628, 628))

    with test("Contour.four_endpoints"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Contour.four_endpoints(contour).must_equal(numpy.array(
              [[ 42, 249],
               [ 49, 583],
               [ 384, 569],
               [ 384, 225]]), numpy.allclose)


    with test("Points.cal_step_points"):
        endpoints = numpy.array([[153,  13],
            [187, 640]], dtype=numpy.int32)
        Points.cal_step_points(endpoints, step_count=9).must_equal(
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
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        four_endpoints = Contour.four_endpoints(contour)
        Points.cal_internal_points(four_endpoints, 9, 9).shape.must_equal((100,2))

    with test("Image.fill_contours"):
        mask = Image.generate_mask((600, 400))
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Image.fill_contours(mask, [contour])
        show_pic(mask)

    # with test("resize"):
    #     image_list = binary_number_to_lists(small_number_path)
    #     image_array = list_to_image_array(image_list)
    #     cur_rect = (7, 9, 18, 18)
    #     sub_image = get_rect_ragion_with_rect(image_array, cur_rect)
    #     sub_image.shape.pp()
    #     # resized_image = cv2.pyrUp(sub_image)
    #     # resized_image.shape.pp()
    #     resized_image = cv2.resize(sub_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    #     # transfer_values(resized_image, {1:255})
    #     resized_image.shape.pp()
    #     save_binary_pic_txt('test_resources/test.txt', sub_image)
    #     save_binary_pic_txt('test_resources/test1.txt', resized_image)
    #     # show_pic(resized_image)


