import Image
import os
import numpy

import cv2
# def show_area(pic_mat, left, top, right, bottom):


#     pass

def matrix_to_image(pic_matrix):
    return Image.fromarray(pic_matrix.A)

def left_white(matrix, treshold):
    row_count, col_count = matrix.shape
    result = 255 - numpy.mat(numpy.zeros(matrix.shape))
    for (i, j), value in numpy.ndenumerate(matrix):
        if value <= treshold:
            result[i, j] = value


    # for i in range(row_count):
    #     for j in range(col_count):
    #         if matrix[i,j] > value:
    #             result[i,j] = matrix[i,j]
    return result

from collections import Counter
def count_horizontal_line(row_indexs, min_line_len=50):
    row_indexs.p()
    return Counter(row_indexs)

def count_vertical_line(indexs):
    pass

if __name__ == '__main__':
    from minitest import *


    pic_path = "../resource/example_pics"
    original_image_name = "original.jpg"
    original_gray_image_name = "original_gray.jpg"

    def get_pic_path(pic_name):
        return os.path.join(pic_path,pic_name)
    
    # with test("to gray"):
    #     original_image = Image.open(get_pic_path(original_image_name))
    #     original_gray_image = original_image.convert('L')
    #     original_gray_image.save(get_pic_path(original_gray_image_name))

    # with test("to array"):
    #     original_gray_image = Image.open(get_pic_path(original_gray_image_name))
    #     gray_matrix = numpy.mat(original_gray_image)
    #     gray_matrix.shape.must_equal((1306, 736))
    #     # save_pic_matrix(gray_matrix[400:700,100:500], get_pic_path("test1.jpg")).show()
    #     # matrix_to_image(gray_matrix[400:700,100:500]).show()
    #     area_matrix = gray_matrix[400:1100,50:700]
    #     threshold = area_matrix.mean()-50
    #     threshold.pp()


    #     area_matrix = left_white(area_matrix, threshold)
    #     area_matrix[0:20,0:200] = 0 # black
    #     area_matrix[0:20,220:420] = threshold
    #     area_matrix[0:20,440:660] = 255 # white

    #     small_area = area_matrix[22:100, 0:100]
    #     black_indexs = numpy.nonzero(area_matrix < 255)
    #     # matrix_to_image(small_area).show()

    #     row_indexs = black_indexs[0].tolist()[0]
    #     count_horizontal_line(row_indexs).pp()
    #     # matrix_to_image(area_matrix).show()
    #     # original_gray_image.show()

    with test("find square"):
        gray_matrix = cv2.imread(get_pic_path(original_gray_image_name), 0)
        area_matrix = gray_matrix[400:1100,50:700]
        cv2.imshow('img', area_matrix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


