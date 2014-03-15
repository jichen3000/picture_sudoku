import numpy
import cv2


BLACK = 0
WHITE = 255
'''
    notice, the point in the pic_arr, like [478, 128], 
    the first one is the column number, 
    the second one is the row number.
    Don't get them reversely.
'''
def find_sudoku_number_binary_arr(pic_arr):
    '''
        Find all numbers from a picture in which there is a soduku puzzle.
        The number form is a binary numpy.array in which number parts are 1, 
        the others are 0.
    '''
    threshold_value = int(pic_arr.mean()*0.7)
    not_use,threshed_arr = cv2.threshold(pic_arr,threshold_value,WHITE,1)
    # not_use.pp()

    square = find_max_square(threshed_arr)

    indexs = cal_squre_area_indexs(square)
    ragion_indexs_arr = cal_split_ragion_indexs_arr(*indexs)

    binary_arr = transfer_values(threshed_arr, {BLACK:0, WHITE:1})

    number_binary_arr = splite_arr_by_ragion_indexs_arr(binary_arr, ragion_indexs_arr)
    return number_binary_arr

def splite_arr_by_ragion_indexs_arr(arr, ragion_indexs_arr):
    return [arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]] 
        for cur_indexs in ragion_indexs_arr]


def find_max_square(threshed_arr):
    '''
        notice: the threshold_value is the key, if it directly impact the binary matrix.
    '''
    contours,not_use = cv2.findContours(threshed_arr.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    def get_approximated_contour(contour):
        perimeter = cv2.arcLength(contour,True)
        return cv2.approxPolyDP(contour,0.01*perimeter,True)

    contours = map(get_approximated_contour, contours)
    squares = filter(is_almost_square, contours)
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
    # area_from_perimeter.pp()
    # real_area.pp()
    if (1-accuracy) * area_from_perimeter < real_area < (1+accuracy) * area_from_perimeter:
        return True
    return False

def cal_squre_area_indexs(contour):
    '''
        calculate the square area values,
        return the start_row_index, end_row_index, start_col_index, end_col_index.
        It can be used like: pic_arr[start_row_index:end_row_index, start_col_index:end_col_index]
        The square must be horizonal.
    '''
    points_count = 4
    flat_arr = contour.flatten('F')
    col_indexs = flat_arr[0: points_count]
    col_indexs.sort()
    row_indexs = flat_arr[points_count: points_count*2]
    row_indexs.sort()
    return row_indexs[1], row_indexs[2], col_indexs[1], col_indexs[2]

def cal_split_ragion_indexs_arr(start_row_index, end_row_index, start_col_index, end_col_index, 
    split_num=9, modified_percent=0.15):
    '''
        firstly row, then col
    '''
    step = int((end_row_index - start_row_index) / split_num)
    modifer = int(step*modified_percent)
    # return [(i,j) for i in range(split_num) for j in range(split_num)]
    result = [(start_row_index+i*step+modifer, start_row_index+(i+1)*step-modifer, 
        start_col_index+j*step+modifer, start_col_index+(j+1)*step-modifer) 
        for i in range(split_num) for j in range(split_num)]
    return result


def transfer_values(arr, rule_hash, is_reverse=False):
    '''
        rule_hash = {0:1, 255:0}
    '''
    if not is_reverse:
        for source, target in rule_hash.items():
            arr[arr==source] = target
    else:
        for target, source in rule_hash.items():
            arr[arr==source] = target
    return arr

def clip_array_by_fixed_size(pic_array, fixed_height=32, fixed_width=32, delta_start_y=3):
    height, width = pic_array.shape
    start_y = int((height - fixed_height)/2)-delta_start_y
    start_x = int((width - fixed_width)/2)
    return pic_array[start_y:start_y+fixed_height, start_x:start_x+fixed_width]


if __name__ == '__main__':
    from minitest import *

    def show_pic(pic_arr):
        cv2.imshow('pic', pic_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ORIGINAL_IMAGE_NAME = '../resource/example_pics/original.jpg'
    # 0 is black, white is 255
    # large is brighter, less is darker.

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

    with test("cal_squre_area_indexs"):
        contour = numpy.array([[[ 671,  421]],
                               [[  78,  426]],
                               [[  85, 1016]],
                               [[ 675, 1012]]])
        cal_squre_area_indexs(contour).must_equal((426, 1012, 85, 671))

    with test("cal_split_ragion_indexs_arr"):
        indexs = (426, 1012, 85, 671)
        ragion_indexs = cal_split_ragion_indexs_arr(*indexs)
        ragion_indexs[0:9].must_equal(
            [(435, 482, 94, 141),
             (435, 482, 159, 206),
             (435, 482, 224, 271),
             (435, 482, 289, 336),
             (435, 482, 354, 401),
             (435, 482, 419, 466),
             (435, 482, 484, 531),
             (435, 482, 549, 596),
             (435, 482, 614, 661)])
        ragion_indexs.size().must_equal(81)

    with test("transfer_values"):
        arr = numpy.array([[255, 255, 255, 0, 255, 255, 255],
                     [255, 255, 255, 0, 255, 255, 255]])
        transfer_values(arr, {0:1, 255:0}).must_equal(
            numpy.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]]), numpy.allclose)

    with test("splite_arr_by_ragion_indexs_arr"):
        cur_indexs=[0,10,0,10]
        splite_arr_by_ragion_indexs_arr(gray_arr, [cur_indexs])[0].must_equal(
            gray_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]], numpy.allclose)

    with test("find_max_square"):
        threshold_value = int(gray_arr.mean()*0.7)
        not_use,current_pic_arr = cv2.threshold(gray_arr,threshold_value,WHITE,1)
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
    #     show_pic(current_pic_arr)

    # with test("show max square in area pic"):
    #     gray_arr.mean().pp()
    #     threshold_value = int(gray_arr.mean()*0.7)
    #     threshold_value.pp()
    #     not_use,current_pic_arr = cv2.threshold(gray_area_arr,threshold_value,WHITE,0)
    #     # current_pic_arr = color_area_arr
    #     area_max_square = find_max_square(current_pic_arr)
    #     cv2.drawContours(current_pic_arr,[area_max_square],-1,(0,255,255),1)
    #     show_pic(current_pic_arr)

    with test("show number pic"):
        threshold_value = int(gray_area_arr.mean()*0.7)
        not_use,current_pic_arr = cv2.threshold(gray_area_arr,threshold_value,WHITE,1)
        area_max_square = find_max_square(current_pic_arr)
        indexs = cal_squre_area_indexs(area_max_square)
        ragion_indexs_arr = cal_split_ragion_indexs_arr(*indexs)
        not_use,current_pic_arr = cv2.threshold(gray_area_arr,WHITE,0,0)
        for cur_indexs in ragion_indexs_arr:
            current_pic_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]] = \
                gray_area_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]]
        # cur_indexs = ragion_indexs_arr[3]
        # current_pic_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]] = \
        #     gray_area_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]]
        # show_pic(current_pic_arr)

    with test("show one number pic"):
        threshold_value = int(gray_arr.mean()*0.7)
        not_use,current_pic_arr = cv2.threshold(gray_area_arr,threshold_value,WHITE,0)
        cur_indexs = ragion_indexs_arr[0]
        current_pic_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]] = \
            gray_area_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]]
        # show_pic(current_pic_arr)
        one_number_pic = gray_area_arr[cur_indexs[0]:cur_indexs[1],cur_indexs[2]:cur_indexs[3]].copy()
        not_use,one_number_pic = cv2.threshold(one_number_pic,threshold_value,WHITE,0)
        # show_pic(one_number_pic)
        # one_number_pic.pp()
        # transfer_values(one_number_pic, {WHITE:0, BLACK:1})
        # numpy.savetxt("test5.dataset",one_number_pic,fmt="%3d")

    with test("find_sudoku_number_binary_arr"):
        number_binary_arr = find_sudoku_number_binary_arr(gray_arr)
        number_5 = number_binary_arr[0]
        black_count = numpy.count_nonzero(number_5)
        white_count = numpy.count_nonzero(1-number_5)
        row_count, col_count = number_5.shape
        (row_count*col_count).must_equal(black_count+white_count)
        black_count.must_equal(91)
        # number_5 = clip_array_by_fixed_size(number_5,delta_start_y=-5)
        numpy.savetxt("test5.dataset",number_5,fmt="%d", delimiter='')
