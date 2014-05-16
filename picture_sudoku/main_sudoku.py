import numpy
import cv2

from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.ragion import Ragion
from picture_sudoku.cv2_helpers.ragion import Ragions

from picture_sudoku.digit_recognition.multiple_svm import MultipleSvm
from picture_sudoku.digit_recognition.rbf_smo import Smo

from picture_sudoku.picture_analyzer import main_analyzer

from picture_sudoku.answer import main_answer

IMG_SIZE = 32
FULL_SIZE = 1024

FONT_RESULT_PATH = '../resource/digit_recognition/font_training_result'

def get_digits(image_path, som_svm, with_ragions=False):
    number_indexs, number_ragions = main_analyzer.extract_number_ragions(image_path)
    # show_number_ragions(number_ragions)
    adjusted_number_ragions = map(adjust_number_ragion, number_ragions)
    number_matrixs = map(transfer_to_digit_matrix, adjusted_number_ragions)
    digits = map(som_svm.dag_classify, number_matrixs)
    digits = tuple(digits)
    if with_ragions:
        return number_indexs, digits, number_ragions
    else:
        return number_indexs, digits

def adjust_number_ragion(the_image):
    element_value = 1
    kernel_size_value = 2*1+1
    kernel = cv2.getStructuringElement(element_value, (kernel_size_value, kernel_size_value))

    # return cv2.dilate(the_image, kernel)
    # closing = erode(dilate(the_image, kernel))    
    return cv2.morphologyEx(the_image, 3, kernel)

def transfer_to_digit_matrix(the_ragion):
    '''
        (1, FULL_SIZE) matrix
    '''
    heighted_ragion = Image.resize_keeping_ratio_by_height(the_ragion, IMG_SIZE)
    standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    return numpy.matrix(standard_ragion.reshape(1, FULL_SIZE))

def answer_quiz_with_pic(pic_file_path):
    smo_svm = MultipleSvm.load_variables(Smo, FONT_RESULT_PATH)
    number_indexs, digits = get_digits(pic_file_path, smo_svm)
    return main_answer.answer_quiz_with_indexs_and_digits(number_indexs, digits)


def answer_quiz_with_point_hash(points_hash):
    return main_answer.answer_quiz_with_point_hash(points_hash)

if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display



    with test(answer_quiz_with_pic):
        i = 1
        pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        answer_quiz_with_pic(pic_file_path).must_equal(
            { 'fixed': 
                   {'0_0': 5, '0_1': 6, '0_3': 8, '0_4': 4, '0_5': 7, 
                    '1_0': 3, '1_2': 9, '1_6': 6, '2_2': 8, '3_1': 1, 
                    '3_4': 8, '3_7': 4, '4_0': 7, '4_1': 9, '4_3': 6, 
                    '4_5': 2, '4_7': 1, '4_8': 8, '5_1': 5, '5_4': 3, 
                    '5_7': 9, '6_6': 2, '7_2': 6, '7_6': 8, '7_8': 7, 
                    '8_3': 3, '8_4': 1, '8_5': 6, '8_7': 5, '8_8': 9},
              'answered':
                   {'0_2': 1, '0_6': 9, '0_7': 2, '0_8': 3, '1_1': 7,
                    '1_3': 5, '1_4': 2, '1_5': 1, '1_7': 8, '1_8': 4,
                    '2_0': 4, '2_1': 2, '2_3': 9, '2_4': 6, '2_5': 3,
                    '2_6': 1, '2_7': 7, '2_8': 5, '3_0': 6, '3_2': 3,
                    '3_3': 7, '3_5': 9, '3_6': 5, '3_8': 2, '4_2': 4,
                    '4_4': 5, '4_6': 3, '5_0': 8, '5_2': 2, '5_3': 1,
                    '5_5': 4, '5_6': 7, '5_8': 6, '6_0': 9, '6_1': 3, 
                    '6_2': 5, '6_3': 4, '6_4': 7, '6_5': 8, '6_7': 6, 
                    '6_8': 1, '7_0': 1, '7_1': 4, '7_3': 2, '7_4': 9, 
                    '7_5': 5, '7_7': 3, '8_0': 2, '8_1': 8, '8_2': 7, 
                    '8_6': 4 }})
        pass
