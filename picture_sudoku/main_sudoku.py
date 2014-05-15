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

def create_smo_svm():
    return MultipleSvm.load_variables(Smo, FONT_RESULT_PATH)

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

    return cv2.dilate(the_image, kernel)

def transfer_to_digit_matrix(the_ragion):
    '''
        (1, FULL_SIZE) matrix
    '''
    heighted_ragion = Image.resize_keeping_ratio_by_height(the_ragion, IMG_SIZE)
    standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    return numpy.matrix(standard_ragion.reshape(1, FULL_SIZE))

def answer_quiz_with_pic(pic_file_path, smo_svm):
    number_indexs, digits = get_digits(pic_file_path, smo_svm)
    return main_answer.answer_quiz_with_indexs_and_digits(number_indexs, digits).ppl()

def answer_quiz_with_point_hash(points_hash):
    return main_answer.answer_quiz_with_point_hash(points_hash)

if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display



    with test(answer_quiz_with_pic):
        i = 1
        pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        answer_quiz_with_pic(pic_file_path, create_smo_svm())
        pass
