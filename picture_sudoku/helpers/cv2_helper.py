import cv2
import numpy
import operator

from cv2_helpers.rect import Rect

BLACK = 0
WHITE = 255


class Contour(object):
    @staticmethod
    def mass_center(contour):
        '''
            Contour.mass_center
        '''
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return (1, 1)
        return ( moments['m10']/moments['m00'], moments['m01']/moments['m00'])

    @staticmethod
    def get_rect_ragion(contour, the_image):
        '''
            Contour.get_rect_ragion
        '''
        return Rect.get_ragion(cv2.boundingRect(contour), the_image)


    @staticmethod
    def vertices_by_min_area(contour):
        '''
            Contour.vertices_by_min_area
        '''
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
            Quadrilateral.vertices
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
    def centroid(quadrilateral):
        '''
            Quadrilateral.centroid 
        '''
        points = quadrilateral.reshape((4,2))
        return numpy.mean(points, axis=0)


    @staticmethod
    def split(quadrilateral,  count_in_row, count_in_col):
        '''
            Quadrilateral.split
            Split to many samll ones.
        '''
        vertices = Quadrilateral.vertices(quadrilateral)
        the_points = Points.cal_internal_points(vertices, count_in_col+1, count_in_row+1)
        return Points.get_quadrilaterals(the_points, count_in_col, count_in_row)

    @staticmethod
    def enlarge(quadrilateral, percent=0.05):
        '''
            Quadrilateral.enlarge
        '''
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
    def get_quadrilaterals(the_points, count_in_col, count_in_row):
        '''
            Points.get_quadrilaterals
            Get the contours which are quadrilaterals.
        '''
        all_points = the_points.reshape((count_in_col+1, count_in_row+1,2))
        quadrilaterals = [Points.to_contour(
                    [all_points[x_index,    y_index], 
                     all_points[x_index,    y_index+1], 
                     all_points[x_index+1,    y_index+1], 
                     all_points[x_index+1,    y_index]]) 
                for x_index in range(count_in_col) for y_index in range(count_in_row)]
        return quadrilaterals


    @staticmethod
    def cal_step_points(two_endpoints_in_line, count):
        '''
            Points.cal_step_points
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
            Points.cal_internal_points
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
        '''
            Points.to_contour
            just support 4 points
        '''
        return numpy.array(points).reshape((4,1,2))

    @staticmethod
    def cal_line_slope(point1, point2):
        '''
            Points.cal_line_slope
            calculate the line's slope which go through the two points
        '''
        x1, y1 = point1
        x2, y2 = point2
        if (x1 - x2) == 0 and (y1 - y2) == 0:
            raise Exception('These two points are same.')
        if (x1 - x2) == 0:
            return float('inf')
        return (float(y1 - y2) / (x1 - x2))


class Ragion(object):
    ''' Actually it is just a part of image. '''                

    @staticmethod
    def fill(the_ragion, target_shape, fill_value=0):
        '''
            Ragion.fill
            Fill the ragion to the size of target_shape.
        '''
        height, width = the_ragion.shape        
        target_height, target_width = target_shape

        if height > target_height and width > target_width:
            return the_ragion

        cur_ragion = the_ragion.copy()
        if target_height > height:
            top_height = (target_height - height) / 2
            bottom_height = target_height - top_height - height

            top_mat = numpy.zeros((top_height, width)) + fill_value
            bottom_mat = numpy.zeros((bottom_height, width)) + fill_value

            cur_ragion = numpy.concatenate((top_mat, cur_ragion, bottom_mat), axis=0)

            # cur_ragion.pp()
        if target_width > width:
            left_width = (target_width - width) / 2
            right_width = target_width - left_width - width

            right_mat = numpy.zeros((target_height, right_width)) + fill_value
            left_mat = numpy.zeros((target_height, left_width)) + fill_value

            cur_ragion = numpy.concatenate((left_mat, cur_ragion, right_mat), axis=1)
        return cur_ragion

class Ragions(object):
    @staticmethod
    def fill_to_same_size(ragions):
        '''
            Ragions.fill_to_same_size
            Fill all ragions to the same size which is the largest in height and width.
        '''
        shapes = map(numpy.shape, ragions)
        largest_height = max(map(operator.itemgetter(0), shapes))
        largest_width = max(map(operator.itemgetter(1), shapes))
        fill_func = lambda ragion: Ragion.fill(ragion, (largest_height, largest_width))
        return map(fill_func, ragions)

    @staticmethod
    def join_same_size(ragions, count_in_row, init_value=BLACK):
        '''
            Ragions.join_same_size
            Join all ragions to a big image.
            The ragions must have same size.
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
        return pic_array

    @staticmethod
    def show_same_size(ragions, count_in_row=9, init_value=BLACK):
        '''
            Ragions.show_same_size
            Show the ragions which have the same size.
        '''
        pic_array = Ragions.join_same_size(ragions, count_in_row, init_value)
        show_pic(pic_array)

    @staticmethod
    def show(ragions, count_in_row=9, init_value=BLACK):
        '''
            Ragions.show
            Show the ragions which could have different size.
        '''
        same_size_ragions = Ragions.fill_to_same_size(ragions)
        count_in_row = min(len(ragions), count_in_row)
        Ragions.show_same_size(same_size_ragions, count_in_row, init_value)


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

    @staticmethod
    def show_rect(the_image, rect):
        sub_image = Rect.get_ragion(rect, the_image)
        show_pic(sub_image)


    @staticmethod
    def show_rects(the_image, rects, color=(0,255,255)):
        contours = map(Rect.to_contour, rects)
        Image.show_contours(the_image, contours, color)

    @staticmethod
    def show_rects_with_color(the_image, rects, color=(0,255,255)):
        Image.show_rects(cv2.cvtColor(the_image, cv2.COLOR_GRAY2BGR), rects, color)



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

    inject(numpy.allclose, 'must_close')

    IMG_SIZE = 32

    # ORIGINAL_IMAGE_NAME = './images/antiqua1.png'
    ORIGINAL_IMAGE_NAME = '../../resource/example_pics/sample01.dataset.jpg'
    small_number_path = '../../resource/test/small_number.dataset'
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

    with test("Ragion.fill"):
        the_ragion = numpy.ones((3,4))
        Ragion.fill(the_ragion, (6,6)).must_close(
            numpy.array(  [[ 0.,  0.,  0.,  0.,  0.,  0.],
                           [ 0.,  1.,  1.,  1.,  1.,  0.],
                           [ 0.,  1.,  1.,  1.,  1.,  0.],
                           [ 0.,  1.,  1.,  1.,  1.,  0.],
                           [ 0.,  0.,  0.,  0.,  0.,  0.],
                           [ 0.,  0.,  0.,  0.,  0.,  0.]]))

    with test("Ragions.fill_to_same_size"):
        ragions = (numpy.ones((3,2)), numpy.ones((2,1)), numpy.ones((2,4)))
        Ragions.fill_to_same_size(ragions).must_close(
            [numpy.array([[ 0.,  1.,  1.,  0.],
                   [ 0.,  1.,  1.,  0.],
                   [ 0.,  1.,  1.,  0.]]),
             numpy.array([[ 0.,  1.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.]]),
             numpy.array([[ 1.,  1.,  1.,  1.],
                   [ 1.,  1.,  1.,  1.],
                   [ 0.,  0.,  0.,  0.]])])

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

    with test("Points.cal_line_slope"):
        points = numpy.array([[-1,1],[1,0]])
        Points.cal_line_slope(points[0], points[1]).must_equal(-0.5)
        points = ([1,1],[1,-1])
        Points.cal_line_slope(points[0], points[1]).must_equal(float('inf'))



    # with test("explain fitLine"):
    #     line = cv2.fitLine(numpy.array([[-1,1],[1,0]]), distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01, aeps=0.01)
    #     line.pp()
    #     the_image = Image.generate_mask((500, 600))
    #     def draw_line(the_image, line):
    #         vx,vy,x,y = line
    #         lefty = int((-x*vy/vx) + y)
    #         righty = int(((the_image.shape[1]-x)*vy/vx)+y)
    #         cv2.line(the_image,(the_image.shape[1]-1,righty),(0,lefty),255,1)

    #     draw_line(the_image, line)
        # Image.show(the_image)

    with test("Quadrilateral.centroid"):
        contour = numpy.array(
              [[[384, 225]],
               [[ 42, 249]],
               [[ 49, 583]],
               [[384, 569]]], dtype=numpy.int32)
        Quadrilateral.centroid(contour).must_equal(
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
        the_image = Image.generate_mask((600, 400))
        # Image.show_contours_with_color(the_image, contours)
        # contours[1:2].pp()
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


        '''
            Ragions.show_same_size
            Show the ragions which have the same size.
        '''
