import numpy
import cv2
import os

from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.ragion import Ragion
from picture_sudoku.cv2_helpers.ragion import Ragions

from picture_sudoku.digit_recognition.multiple_svm import MultipleSvm
from picture_sudoku.digit_recognition.rbf_smo import Smo
from picture_sudoku.digit_recognition import data_file_helper


from picture_sudoku.picture_analyzer import main_analyzer

from picture_sudoku.answer import main_answer
from picture_sudoku.helpers.common import Resource, OtherResource

IMG_SIZE = data_file_helper.IMG_SIZE
FULL_SIZE = data_file_helper.FULL_SIZE

# RESULT_PATH = data_file_helper.FONT_RESULT_PATH
RESULT_PATH = data_file_helper.SUPPLEMENT_RESULT_PATH

import logging
logger = logging.getLogger(__name__)

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILURE = "FAILURE"
RESULT_STATUS = "status"
RESULT_ERROR = "error"
RESULT_FILENAME = "pic_file_name"
RESULT_FIXED = "fixed"
RESULT_ANSWERED = "answered"
ERROR_CANNOT_ANSWER = "Sorry, cannot answer it, since this puzzle may not follow the rules of Sudoku!"



def get_digits(image_path, som_svm, with_ragions=False):
    number_indexs, number_ragions = main_analyzer.extract_number_ragions(image_path)
    # Display.binary_image(number_ragions[1])
    # number_indexs.pl()
    # issue_ragion = number_ragions[number_indexs.index(20)]
    # Image.save_to_txt(issue_ragion,Resource.get_path('test/binary_image_6_8_03.dataset'))
    # show_number_ragions(number_ragions)
    # som_svm.dag_classify(transfer_to_digit_matrix(adjust_number_ragion(issue_ragion))).pl()
    # Image.save_to_txt(adjust_number_ragion(issue_ragion), 
    #     Resource.get_path('test/sample_16_08_30_ori.dataset'))
    # Display.binary_ragions(number_ragions)

    # it will adjust number ragion
    # adjusted_number_ragions = map(adjust_number_ragion, number_ragions)
    # number_matrixs = map(transfer_to_digit_matrix, adjusted_number_ragions)
    # it remove the adjusted_number_ragions
    number_matrixs = map(transfer_to_digit_matrix, number_ragions)
    digits = map(som_svm.dag_classify, number_matrixs)

    # special handling for number 8
    # digits = map(Ragion.review_classified_number_ragion_for_8, number_ragions, digits)
    # number_indexs.pl()
    # digits.pl()
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
    # heighted_ragion = Image.resize_keeping_ratio_by_height(the_ragion, IMG_SIZE)
    # standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    # INTER_CUBIC INTER_AREA INTER_LINEAR
    # the_ragion.shape.p()
    standard_ragion = resize_to_cell_size(the_ragion, interpolation = cv2.INTER_AREA)
    # flag_test()
    # Display.binary_image(standard_ragion)
    # standard_ragion.shape.p()
    return numpy.matrix(standard_ragion.reshape(1, FULL_SIZE))

def resize_to_cell_size(the_ragion, interpolation = cv2.INTER_AREA):
    # the_ragion.shape.pl()
    heighted_ragion = Image.resize_keeping_ratio_by_fixed_length(the_ragion, IMG_SIZE, interpolation)
    # heighted_ragion.shape.pl()
    standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    # standard_ragion.shape.pl()
    return standard_ragion


global_smo_svm = None
def answer_quiz_with_pic(pic_file_path):
    def intent_func():
        global global_smo_svm
        if global_smo_svm==None:
            # global_smo_svm.ppl()
            logger.info("The path of training result: {0}".format(RESULT_PATH))
            global_smo_svm = MultipleSvm.load_variables(Smo, RESULT_PATH)
        number_indexs, digits = get_digits(pic_file_path, global_smo_svm)
        return main_answer.answer_quiz_with_indexs_and_digits(number_indexs, digits)
    result = answer_common(intent_func)
    result[RESULT_FILENAME] = os.path.basename(pic_file_path)
    return result

def answer_quiz_with_point_hash(points_hash):
    def intent_func():
        return {RESULT_ANSWERED: main_answer.answer_quiz_with_point_hash(points_hash)}
    return answer_common(intent_func)

def answer_common(the_func):
    try:
        result=the_func()
        if (result[RESULT_ANSWERED]):
            result[RESULT_STATUS] = STATUS_SUCCESS
        else:
            # remove key for java
            result.pop(RESULT_ANSWERED)
            result[RESULT_STATUS] = STATUS_FAILURE
            result[RESULT_ERROR] = ERROR_CANNOT_ANSWER
    except Exception, e:
        import traceback
        print(traceback.format_exc())
        import sys
        sys.stdout.flush()
        result = {RESULT_STATUS:STATUS_FAILURE, 
                RESULT_ERROR:e.message}
    return result

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
                    '8_6': 4 },
              'pic_file_name': 'sample01.dataset.jpg',
              'status': 'SUCCESS'})
        pass

    with test("answer_quiz_with_pic with puzzle which cannot answer it"):
        i = 7
        pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        answer_quiz_with_pic(pic_file_path).must_equal(
            {'pic_file_name': 'sample07.dataset.jpg', 
             'status': 'FAILURE', 
             'error': ERROR_CANNOT_ANSWER,
             'fixed': 
                    {'8_0': 9, '1_0': 5, '1_7': 6, '8_7': 4, '8_4': 4, 
                     '1_4': 2, '7_8': 6, '1_1': 7, '8_1': 1, '0_7': 1, 
                     '7_7': 4, '7_1': 5, '0_4': 9, '0_0': 8, '0_1': 2, 
                     '3_3': 4, '3_5': 5, '3_4': 8, '3_8': 1, '7_0': 3, 
                     '5_3': 9, '5_0': 1, '8_8': 8, '5_5': 3, '5_4': 7, 
                     '1_8': 4, '0_8': 3, '7_4': 1, '5_8': 2}
            })

    with test("answer_quiz_with_pic with false picture, it will report error"):
        i = 1
        pic_file_path = '../resource/example_pics/false'+str(i).zfill(2)+'.dataset.jpg'
        answer_quiz_with_pic(pic_file_path).must_equal(
            {'error': "'NoneType' object has no attribute '__getitem__'",
             'pic_file_name': 'false01.dataset.jpg',
             'status': 'FAILURE'})

    with test(answer_quiz_with_point_hash):
        point_hash = {  u'0_0': 5, u'0_1': 6, u'0_3': 8, u'0_4': 4, u'0_5': 7, 
                        u'1_0': 3, u'1_2': 9, u'1_6': 6, u'2_2': 8, u'3_1': 1, 
                        u'3_4': 8, u'3_7': 4, u'4_0': 7, u'4_1': 9, u'4_3': 6, 
                        u'4_5': 2, u'4_7': 1, u'4_8': 8, u'5_1': 5, u'5_4': 3, 
                        u'5_7': 9, u'6_6': 2, u'7_2': 6, u'7_6': 8, u'7_8': 7, 
                        u'8_3': 3, u'8_4': 1, u'8_5': 6, u'8_7': 5, u'8_8': 9 }
        answer_quiz_with_point_hash(point_hash).must_equal(
            { 'answered':
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
                    '8_6': 4 },
              'status': 'SUCCESS'})

    with test("answer_quiz_with_point_hash with puzzle which cannot answer it"):
        point_hash = {  u'0_0': 5, u'0_1': 6, u'0_3': 8, u'0_4': 4, u'0_5': 7, 
                        u'1_0': 3, u'1_2': 9, u'1_6': 6, u'2_2': 8, u'3_1': 1, 
                        u'3_4': 8, u'3_7': 4, u'4_0': 7, u'4_1': 9, u'4_3': 6, 
                        u'4_5': 2, u'4_7': 1, u'4_8': 8, u'5_1': 5, u'5_4': 3, 
                        u'5_7': 9, u'6_6': 2, u'7_2': 6, u'7_6': 8, u'7_8': 7, 
                        u'8_3': 3, u'8_4': 1, u'8_5': 5, u'8_7': 5, u'8_8': 9 }
        answer_quiz_with_point_hash(point_hash).must_equal(
            {'error': ERROR_CANNOT_ANSWER, 'status': 'FAILURE'})

