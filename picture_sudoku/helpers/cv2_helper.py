'''
    Test in the codes/python/projects/font_number_binary/cv2_helper.py
'''


import cv2
import numpy



BLACK = 0
WHITE = 255

class Rect(object):
    '''
        x, y, width, height = rect
    '''
    @staticmethod
    def modify_to_ratio(rect, shape):
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

    @staticmethod
    def split_to_rects(rect, split_x_num, split_y_num):
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

    @staticmethod
    def get_ragion(rect, the_image):
        x, y, width, height = rect
        return the_image[y:y+height,x:x+width]

    @staticmethod
    def vertices(rect):
        x, y, width, height = rect
        return numpy.array([(x,         y),
                            (x,         y+height-1),
                            (x+width-1, y+height-1),
                            (x+width-1, y)])

    @staticmethod
    def to_contour(rect):
        return Rect.vertices(rect).reshape((4,1,2))


class Contour(object):
    @staticmethod
    def mass_center(contour):
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return (1, 1)
        return ( moments['m10']/moments['m00'], moments['m01']/moments['m00'])

    @staticmethod
    def get_rect_ragion(contour, the_image):
        return Rect.get_ragion(cv2.boundingRect(contour), the_image)


    @staticmethod
    def vertices_by_min_area(contour):
        ''' another way to get the rect of a contour
            just like boundingRect
        '''
        rect = cv2.minAreaRect(contour)
        points = cv2.cv.BoxPoints(rect)
        points = numpy.int0(points)
        return points


class Quadrilateral(Contour):
    @staticmethod
    def vertices(quadrilateral):
        '''
            It will adjust points to top_left, bottom_left, bottom_right, top_right.
        '''
        points = quadrilateral.reshape((4,2))
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

    @staticmethod
    def center(quadrilateral):
        points = quadrilateral.reshape((4,2))
        return numpy.mean(points, axis=0)


    @staticmethod
    def split(quadrilateral,  count_in_row, count_in_col):
        '''
            Split to many samll ones.
        '''
        vertices = Quadrilateral.vertices(quadrilateral)
        all_points = Points.cal_internal_points(vertices, count_in_col+1, count_in_row+1)
        all_points = all_points.reshape((count_in_col+1, count_in_row+1,2))
        # all_points.shape.pp()
        # all_points[3,2].pp()
        # return [all_points[0, 0], all_points[0, 1], all_points[1, 1], all_points[1,0] ]
        quadrilaterals = []
        for x_index in range(count_in_col):
            for y_index in range(count_in_row):
                quadrilaterals.append(Points.to_contour(
                    [all_points[x_index,    y_index], 
                     all_points[x_index,    y_index+1], 
                     all_points[x_index+1,    y_index+1], 
                     all_points[x_index+1,    y_index]]))
        return quadrilaterals

    @staticmethod
    def enlarge(quadrilateral, percent=0.05):
        x, y, width, height = cv2.boundingRect(quadrilateral)
        x_step = int(width * percent)
        y_step = int(height * percent)
        top_left, bottom_left, bottom_right, top_right = \
                Quadrilateral.vertices(quadrilateral)
        top_left=(top_left[0]-x_step, top_left[1]-y_step)
        top_right=(top_right[0]+x_step, top_right[1]-y_step)
        bottom_right=(bottom_right[0]+x_step, bottom_right[1]+y_step)
        bottom_left=(bottom_left[0]-x_step, bottom_left[1]+y_step)
        return Points.to_contour((top_left, top_right, bottom_right, bottom_left))





class Points(object):
    @staticmethod
    def cal_step_points(two_endpoints_in_line, count):
        '''
            two_endpoints_in_line are the two point of a line.
            count will be the points on that line between the two endpoints.
        '''
        step_distance = numpy.float32(two_endpoints_in_line[-1] - two_endpoints_in_line[0]) / (count-1)
        points_list = [(two_endpoints_in_line[0] + index * step_distance)
                for index in range(count)]
        return numpy.array(numpy.int32(points_list)).reshape((count,2))
    
    @staticmethod
    def cal_internal_points(vertices, row_count, col_count):
        '''
            The point's order will be from left to right then top to bottom.
        '''
        left_points = Points.cal_step_points(vertices[0:2], row_count)
        rifht_points = Points.cal_step_points(vertices[::-1][0:2], row_count)
        internal_points = [] 
        for left_point, right_point in zip(left_points, rifht_points):
            # (left_point, right_point).pp()
            endpoints = numpy.append(left_point, right_point).reshape(2,2)
            internal_points.append(Points.cal_step_points(endpoints, col_count))

        return numpy.array(internal_points).reshape((col_count)*(row_count),2)

    @staticmethod
    def to_contour(points):
        return numpy.array(points).reshape((4,1,2))



class Image(object):
    @staticmethod
    def center_point(the_image):
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
        return numpy.savetxt(the_image, file_name, fmt='%d', delimiter='')

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
    def show(the_image, image_name='image'):
        cv2.imshow(image_name, the_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_contours_with_color(the_image, contours, color=(0,255,255)):
        Image.show_contours(cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR), contours, color)

    @staticmethod
    def show_contours(the_image, contours, color=(0,255,255)):
        cv2.drawContours(the_image, contours, -1, color ,1)
        # points = map(Contour.mass_center, contours)
        # for point in numpy.int32(points):
        #     cv2.circle(the_image,tuple(point),2,color,-1)
        Image.show(the_image)

    @staticmethod
    def show_points_with_color(the_image, points, color=(0,255,0)):
        int_points = numpy.int32(points)
        color_image = cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR)
        for point in int_points:
            cv2.circle(color_image,tuple(point),2,color,-1)
        Image.show(color_image)



def threshold_white_with_mean_percent(gray_pic, mean_percent=0.7):
    return Image.threshold_white_with_mean_percent(gray_pic, mean_percent)

def find_contours(threshed_pic_array, filter_func = None, accuracy_percent_with_perimeter=0.01):
    return Image.find_contours(threshed_pic_array, filter_func, accuracy_percent_with_perimeter)

def clip_array_by_x_y_count(pic_array, clip_x_count=2, clip_y_count=2):
    return Image.clip_by_x_y_count(pic_array, clip_x_count, clip_y_count)

def clip_array_by_percent(pic_array, percent=0.1):
    return Image.clip_by_percent(pic_array, percent)

def clip_array_by_four_percent(pic_array, 
        top_percent=0.1, bottom_percent=0.1, left_percent=0.1, right_percent=0.1):
    return Image.clip_by_four_percent(pic_array, 
        top_percent, bottom_percent, left_percent, right_percent)

def save_binary_pic_txt(file_name, binary_pic):
    return Image.save_to_txt(binary_pic, file_name)


def is_not_empty_pic(pic_array):
    return Image.is_not_empty(pic_array)


def cal_nonzero_rect(pic_array):
    return Image.cal_nonzero_rect(pic_array)


def cal_nonzero_rect_as_pic_ratio(pic_array):
    return Image.cal_nonzero_rect_keeping_ratio(pic_array)



def show_pic(the_image):
    Image.show(the_image)

def show_contours_in_pic(pic_array, contours, color=(0,255,255)):
    Image.show_contours(pic_array, contours, color)

def show_contours_in_color_pic(pic_array, contours, color=(0,255,255)):
    Image.show_contours_with_color(pic_array, contours, color)

def show_rects_in_pic(pic_array, rects, color=(0,255,255)):
    contours = map(Rect.to_contour, rects)
    show_contours_in_pic(pic_array, contours, color)

def show_rects_in_color_pic(pic_array, rects, color=(0,255,255)):
    show_rects_in_pic(cv2.cvtColor(pic_array, cv2.COLOR_GRAY2BGR), rects, color)

def show_rect(pic_array, rect):
    sub_pic_array = Rect.get_ragion(rect, pic_array)
    show_pic(sub_pic_array)

def show_points_in_color_pic(pic_array, points):
    Image.show_points_with_color(pic_array, points)

def show_same_size_ragions_as_pic(ragions, count_in_row=9, init_value=BLACK):
    pic_array = join_ragions_as_pic(ragions, count_in_row, init_value)
    show_pic(pic_array)

def modify_rect_as_pic_ratio(rect, shape):
    return Rect.modify_to_ratio(rect, shape)

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

''' methods not using '''
# def clip_array_by_fixed_size(pic_array, fixed_height=32, fixed_width=32, delta_start_y=3):
#     height, width = pic_array.shape
#     if fixed_height > height or fixed_width > width:
#         raise Exception("out of picture's border %d > %d or %d > %d" %(fixed_height, height, fixed_width, width))
#     start_y = int((height - fixed_height)/2)-delta_start_y
#     if start_y < 0:
#         start_y = 0
#     start_x = int((width - fixed_width)/2)
#     return pic_array[start_y:start_y+fixed_height, start_x:start_x+fixed_width]


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

    with test("Contour.get_rect_ragion"):
        contour = numpy.array([[[ 171,  21]],
                               [[  18,  26]],
                               [[  25, 216]],
                               [[ 175, 212]]])
        '''
            notice the result will be different with different picture.
        '''
        cur_area_pic1 = Contour.get_rect_ragion(contour, gray_pic)
        cur_area_pic1.shape.must_equal((196, 158))
        cur_area_pic1[0,0].must_equal(gray_pic[18,21])

        rect = (18, 21, 158, 196)
        cv2.boundingRect(contour).must_equal(rect)
        cur_area_pic2 = Rect.get_ragion(rect, gray_pic)
        cur_area_pic2.shape.must_equal((196, 158))
        cur_area_pic2[0,0].must_equal(gray_pic[18,21])
        ''' uncomment the below, you can see the consequence in a picture. '''
        # show_pic(cur_area_pic1)    

    with test("Rect.split_to_rects"):
        contour = numpy.array(
              [[[ 16, 139]],
               [[ 16, 225]],
               [[478, 225]],
               [[478, 139]]])
        ragion_rects = Rect.split_to_rects(cv2.boundingRect(contour), 10, 1)
        ragion_contours = map(Rect.to_contour, ragion_rects)
        ragion_contours.size().must_equal(10)
        ''' uncomment the below, you can see the consequence in a picture. '''        
        # show_contours_in_pic(color_pic, ragion_contours)
        pass

    with test("Rect.to_contour"):
        contour = numpy.array([[[ 18,  21]],
                               [[ 18, 216]],
                               [[175, 216]],
                               [[175,  21]]])
        rect = (18, 21, 158, 196)
        Rect.to_contour(rect).must_equal(contour, numpy.allclose)

    with test("Rect.vertices"):
        rect = (18, 21, 158, 196)
        Rect.vertices(rect).must_equal(
            numpy.array([  [ 18,  21],
                           [ 18, 216],
                           [175, 216],
                           [175,  21]]), numpy.allclose)


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

    with test("cal_nonzero_rect"):
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

    with test("Rect.modify_to_ratio"):
        cur_rect = (10, 9, 12, 18)
        Rect.modify_to_ratio(cur_rect, (IMG_SIZE, IMG_SIZE)).must_equal(
            (7, 9, 18, 18))

    with test("Image.cal_nonzero_rect_keeping_ratio"):
        image_list = binary_number_to_lists(small_number_path)
        image_array = list_to_image_array(image_list)
        Image.cal_nonzero_rect_keeping_ratio(image_array).must_equal((7, 9, 18, 18))

    with test("Image.is_not_empty"):
        arr = numpy.zeros((3,3))
        Image.is_not_empty(arr).must_equal(False)
        arr[1,1] = 1 
        Image.is_not_empty(arr).must_equal(True)

    with test("Rect.adjust_to_minimum"):
        rect = (153, 13, 668, 628)
        Rect.adjust_to_minimum(rect).must_equal((153, 13, 628, 628))

    with test("Quadrilateral.vertices"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Quadrilateral.vertices(contour).must_equal(numpy.array(
              [[ 42, 249],
               [ 49, 583],
               [ 384, 569],
               [ 384, 225]]), numpy.allclose)

    with test("Quadrilateral.center"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Quadrilateral.center(contour).must_equal(
            numpy.array(
              [ 214.75,  406.5 ]), numpy.allclose)


    with test("Quadrilateral.split"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        contours = Quadrilateral.split(contour,2,3)
        contours.size().must_equal(6)
        contours[1].must_equal(numpy.array(
              [[[213, 237]],
               [[384, 225]],
               [[384, 339]],
               [[214, 349]]]), numpy.allclose)

        ''' show '''
        # the_image = Image.generate_mask((600, 400))
        # Image.show_contours_with_color(the_image, contours)
        # Image.show_contours_with_color(the_image, contours[1:2])

    with test("Quadrilateral.enlarge"):
        quadrilateral = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        percent = 0.05
        enlarged_quadrilateral = Quadrilateral.enlarge(quadrilateral, percent)
        (cv2.arcLength(enlarged_quadrilateral, True) > cv2.arcLength(quadrilateral, True) * (1+percent*2)
            ).must_equal(True)

        ''' show '''
        # the_image = Image.generate_mask((700, 500))
        # Image.show_contours_with_color(the_image, [quadrilateral, enlarged_quadrilateral])



    with test("Contour.mass_center"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Contour.mass_center(contour).must_equal(
            (215.54283010213263, 405.8626870351339))
        # contour = contour.reshape((4,2))
        # Contour.mass_center(contour).must_equal(
        #     (215.54283010213263, 405.8626870351339))


    with test("Points.cal_step_points"):
        endpoints = numpy.array([[153,  13],
            [187, 640]], dtype=numpy.int32)
        Points.cal_step_points(endpoints, 10).must_equal(
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
        vertices = numpy.array(
              [[ 42, 249],
               [ 49, 583],
               [ 384, 569],
               [ 384, 225]], dtype=numpy.int32)
        points = Points.cal_internal_points(vertices, row_count=4, col_count=3)
        points.must_equal(numpy.array([[ 42, 249],
                                       [213, 237],
                                       [384, 225],
                                       [ 44, 360],
                                       [214, 349],
                                       [384, 339],
                                       [ 46, 471],
                                       [215, 462],
                                       [384, 454],
                                       [ 49, 583],
                                       [216, 576],
                                       [384, 569]], dtype=numpy.int32), numpy.allclose)
        ''' show '''
        # the_image = Image.generate_mask((600, 400))
        # Image.show_points_with_color(the_image, points)


    with test("Image.fill_contours"):
        mask = Image.generate_mask((600, 400))
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Image.fill_contours(mask, [contour])

    # with test("resize"):
    #     image_list = binary_number_to_lists(small_number_path)
    #     image_array = list_to_image_array(image_list)
    #     cur_rect = (7, 9, 18, 18)
    #     sub_image = Rect.get_ragion(cur_rect, image_array)
    #     sub_image.shape.pp()
    #     # resized_image = cv2.pyrUp(sub_image)
    #     # resized_image.shape.pp()
    #     resized_image = cv2.resize(sub_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    #     # transfer_values(resized_image, {1:255})
    #     resized_image.shape.pp()
    #     Image.save_to_txt('test_resources/test.txt', sub_image)
    #     Image.save_to_txt('test_resources/test1.txt', resized_image)
    #     # show_pic(resized_image)


