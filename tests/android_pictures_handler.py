import cv2
import numpy
from picture_sudoku.picture_analyzer import main_analyzer
from picture_sudoku.picture_analyzer import nonzero_rect
from picture_sudoku.digit_recognition.multiple_svm import MultipleSvm
from picture_sudoku.digit_recognition.rbf_smo import Smo
from picture_sudoku import main_sudoku
from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.ragion import Ragion
from picture_sudoku.cv2_helpers.rect import Rect
from picture_sudoku.digit_recognition import multiple_svm


IMG_SIZE = 32
FULL_SIZE = 1024

def transfer_to_digit_matrix(the_ragion):
    '''
        (1, FULL_SIZE) matrix
    '''
    # heighted_ragion = Image.resize_keeping_ratio_by_height(the_ragion, IMG_SIZE)
    # standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    standard_ragion = resize_to_cell_size(the_ragion)
    return numpy.matrix(standard_ragion.reshape(1, FULL_SIZE))

def reshape_to_one_row(the_ragion):
    height, width = the_ragion.shape
    return numpy.matrix(the_ragion.reshape(1, height*width))


def resize_to_cell_size(the_ragion):
    heighted_ragion = resize_keeping_ratio_by_height(the_ragion, IMG_SIZE)
    standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    return standard_ragion

def resize_keeping_ratio_by_height(the_image, height=700):
    width = float(height) / the_image.shape[0]
    dim = (int(the_image.shape[1] * width), height)
    return cv2.resize(the_image, dim, interpolation = cv2.INTER_AREA)
    # return cv2.resize(the_image, dim, interpolation = cv2.INTER_LINEAR)

def adjust_number_ragion(the_image):
    element_value = 1
    kernel_size_value = 2*1+1
    # kernel_size_value = 2*1+1
    kernel = cv2.getStructuringElement(element_value, (kernel_size_value, kernel_size_value))

    # return cv2.erode(the_image, kernel)
    return cv2.erode(cv2.dilate(the_image, kernel),kernel)    
    # return cv2.morphologyEx(the_image, 3, kernel)

def review_classified_number_ragion_for_8(the_ragion, the_digit):
    '''
        change the digit which has been recognized as 8,
        but actually is the other ones, like 6. 
    '''
    if the_digit != 8 :
        return the_digit
    height, width = the_ragion.shape
    def review_for_6():
        # for 6, check the top right part
        end_y = height / 2
        start_y = height / 4
        start_x = width / 2
        end_x = width
        for y_index in range(end_y+1, start_y, -1):
            for x_index in range(start_x, end_x):
                if(the_ragion[y_index, x_index] > 0):
                    return 6
        return 8
    return review_for_6()


if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.helpers.common import Resource, OtherResource
    from picture_sudoku.helpers import numpy_helper
    from picture_sudoku.digit_recognition import data_file_helper

    import __builtin__

    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    __builtin__.Display = Display
    # with test("for issue 6 to 8"):
    #     pic_file_path = 'for_issues/colin_demo.jpg'
    #     # pic_file_path = 'for_issues/cannot_recognize.jpg'
    #     # pic_file_path = 'for_issues/cannot_recognize_02.jpg'
    #     image_path = Resource.get_path(pic_file_path)
    #     gray_image = cv2.imread(image_path, 0)
    #     # Display.image(gray_image)
    #     number_indexs, number_ragions = main_analyzer.extract_number_ragions(image_path)
    #     number_ragions_255 = map(numpy_helper.transfer_1to255, number_ragions)
        # main_analyzer.show_all(gray_image, zip(number_indexs, number_ragions_255))
    #     issue_ragion = main_analyzer.extract_specified_number_ragion(image_path, 3, 3)
    #     Image.save_to_txt(issue_ragion,Resource.get_path('test/binary_image_6_8_01.dataset'))
    #     # Display.binary_image(issue_ragion)
    #     # numpy.save(Resource.get_path('test/binary_image.npy'), issue_ragion)
    #     # Display.ragions_binary(number_ragions)
    #     # number_indexs.ppl()
    #     # Display.image(numpy_helper.transfer_values_quickly(number_ragions[0],{1:255}))
    #     # number_ragions[0].shape.ppl()

    #     # adjusted_ragion =  main_sudoku.adjust_number_ragion(issue_ragion)
    #     # transfered_ragion = main_sudoku.transfer_to_digit_matrix(adjusted_ragion)
    #     # # Display.binary_image(adjusted_ragion)
    #     # Display.binary_image(main_sudoku.resize_to_cell_size(issue_ragion))
    #     # font_result_path = '../other_resource/font_training_result'
    #     # som_svm = MultipleSvm.load_variables(Smo, font_result_path)
    #     # som_svm.dag_classify(transfered_ragion).pl()

    # with test("just for issue 6 to 8"):
    #     # file_path = 'test/binary_image_6_8.dataset'
    #     # file_path = 'test/sample_15_08_80_original.dataset'
    #     file_path = 'test/sample_16_06_30.dataset'
    #     # file_path = 'test/sample_17_06_52.dataset'
    #     issue_ragion = Image.load_from_txt(Resource.get_path(file_path))
    #     # Display.binary_image(issue_ragion)
    #     # adjusted_ragion =  adjust_number_ragion(issue_ragion)
    #     # Display.binary_image(adjusted_ragion)
    #     # transfered_ragion = transfer_to_digit_matrix(adjusted_ragion)
    #     # transfered_ragion = transfer_to_digit_matrix(issue_ragion)
    #     transfered_ragion = main_sudoku.transfer_to_digit_matrix(issue_ragion)
    #     # Image.save_to_txt(resize_to_cell_size(issue_ragion), 
    #     #     Resource.get_path('test/sample_17_06_30_tra.dataset'))
    #     # Image.save_to_txt(adjust_number_ragion(issue_ragion), 
    #     #     Resource.get_path('test/sample_17_06_30_adj02.dataset'))
    #     font_result_path = OtherResource.get_path('font_training_result')

    #     som_svm = MultipleSvm.load_variables(Smo, font_result_path)
    #     # adjusted_number_ragions = map(adjust_number_ragion, number_ragions)
    #     # number_matrixs = map(transfer_to_digit_matrix, adjusted_number_ragions)
    #     # digits = map(som_svm.dag_classify, number_matrixs)
    #     # som_svm.dag_classify(transfered_ragion).pl()

    #     # resized_ragion = resize_to_cell_size(issue_ragion)
    #     # adjusted_ragion =  main_sudoku.adjust_number_ragion(resized_ragion)
    #     # transfered_ragion = numpy.matrix(adjusted_ragion.reshape(1, FULL_SIZE))
    #     # Image.save_to_txt(main_sudoku.adjust_number_ragion(adjusted_ragion), 
        # #     Resource.get_path('test/sample_17_06_30_adj01.dataset'))

        # the_digit = som_svm.dag_classify(transfered_ragion).pl()
        # review_classified_number_ragion_for_8(issue_ragion, the_digit).pl()


    # with test("for issue too big"):
    #     # image_path = Resource.get_path('for_issues/cannot_recognize_02.jpg')
    #     image_path = Resource.get_path('example_pics/false01.dataset.jpg')
    #     # image_path = Resource.get_path('example_pics/sample01.dataset.jpg')
    #     # # # number_indexs, number_ragions = main_analyzer.extract_number_ragions(image_path)
    #     # gray_image = cv2.imread(image_path, 0)
    #     # gray_image = cv2.GaussianBlur(gray_image,ksize=(5,5), sigmaX=0)

    #     # threshed_image = cv2.adaptiveThreshold(gray_image, 255,
    #     #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

    #     # # ''' 
    #     # #     It's very depend on the threshed_image.
    #     # # '''
    #     # max_contour = find_max_contour(threshed_image)
    #     # Display.contours(gray_image, [max_contour])
    #     main_sudoku.answer_quiz_with_pic(image_path).pl()

    #     pass

    # with test("for issue Unrecognized or unsupported array type in function cvGetMat"):
    #     image_path = Resource.get_path('for_issues/colin_demo.jpg')
    #     # image_path = Resource.get_path('for_issues/cannot_recognize_02.jpg')
    #     main_sudoku.answer_quiz_with_pic(image_path).pl()
    #     # gray_image = cv2.imread(image_path, 0)
    #     # Display.image(gray_image)
    #     # /Users/colin/work/picture_sudoku/other_resource/font_training_result

    # with test("for having more than two border lines"):
    #     # image_path = Resource.get_path('example_pics/sample11.dataset.jpg')
    #     # image_path = Resource.get_path('example_pics/sample08.dataset.jpg')
    #     # image_path = Resource.get_path('example_pics/sample16.dataset.jpg')
    #     image_path = Resource.get_path('for_issues/cannot_recognize.jpg')
    #     main_sudoku.answer_quiz_with_pic(image_path).pl()
    #     # gray_image = cv2.imread(image_path, 0)
    #     # Display.image(gray_image)

    with test("get clear number ragion"):
        som_svm = MultipleSvm.load_variables(Smo, data_file_helper.SUPPLEMENT_RESULT_PATH)
        file_path = Resource.get_test_path('sample_15_null_38_image.jpg')
        the_ragion = cv2.imread(file_path, 0)
        # the_ragion.mean().ppl()
        # the_ragion.ppl()
        # thresholded_ragion = Image.threshold_white_with_mean_percent(the_ragion, 0.8)
        # thresholded_ragion.ppl()
        # Display.image(thresholded_ragion)
        file_path = Resource.get_test_path('sample_15_square.jpg')
        square_ragion = cv2.imread(file_path, 0)
        # square_ragion.mean().ppl()

        threshold_value = Ragion.cal_threshold_value(the_ragion, square_ragion, 0.69)
        thresholded_ragion = Image.threshold_white(the_ragion, threshold_value)
        # thresholded_ragion = cv2.adaptiveThreshold(the_ragion, 255,
        #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)
        cell_rect = nonzero_rect.analyze_from_center(thresholded_ragion)
        if cell_rect:
            cell_ragion = Rect.get_ragion(cell_rect, thresholded_ragion)
        cell_rect.pl()
        # Display.image(cell_ragion)


        file_path = Resource.get_test_path('sample_19_07_05_image.jpg')
        the_ragion = cv2.imread(file_path, 0)
        # the_ragion.mean().ppl()

        file_path = Resource.get_test_path('sample_19_square.jpg')
        square_ragion = cv2.imread(file_path, 0)
        # square_ragion.mean().ppl()

        threshold_value = Ragion.cal_threshold_value(the_ragion, square_ragion, 0.8)
        thresholded_ragion = Image.threshold_white(the_ragion, threshold_value)
        cell_rect = nonzero_rect.analyze_from_center(thresholded_ragion)
        if cell_rect:
            cell_ragion = Rect.get_ragion(cell_rect, thresholded_ragion)
        number_ragion = numpy_helper.transfer_255to1(cell_ragion)
        number_matrix = main_sudoku.transfer_to_digit_matrix(number_ragion)
        som_svm.dag_classify(number_matrix).pl()
        # thresholded_ragion.ppl()
        # Display.image(thresholded_ragion)

