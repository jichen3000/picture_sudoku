import cv2
import numpy
from picture_sudoku.picture_analyzer import main_analyzer
from picture_sudoku.digit_recognition.multiple_svm import MultipleSvm
from picture_sudoku.digit_recognition.rbf_smo import Smo
from picture_sudoku import main_sudoku
from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.digit_recognition import multiple_svm


if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.helpers.common import Resource, OtherResource
    from picture_sudoku.helpers import numpy_helper

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
    #     file_path = 'test/binary_image_6_8_03.dataset'
    #     issue_ragion = Image.load_from_txt(Resource.get_path(file_path))
    #     # Display.binary_image(issue_ragion)
    #     adjusted_ragion =  main_sudoku.adjust_number_ragion(issue_ragion)
    #     # Display.binary_image(adjusted_ragion)
    #     transfered_ragion = main_sudoku.transfer_to_digit_matrix(adjusted_ragion)
    #     Image.save_to_txt(adjusted_ragion, Resource.get_path('for_issues/binary_image_6_8_dd01.dataset'))
    #     font_result_path = OtherResource.get_path('font_training_result')
    #     font_result_path.ppl()
    #     som_svm = MultipleSvm.load_variables(Smo, font_result_path)
    #     # adjusted_number_ragions = map(adjust_number_ragion, number_ragions)
    #     # number_matrixs = map(transfer_to_digit_matrix, adjusted_number_ragions)
    #     # digits = map(som_svm.dag_classify, number_matrixs)
    #     som_svm.dag_classify(transfered_ragion).pl()


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

    with test("for having more than two border lines"):
        '''
            when I use , it will report two border lines.
            5,6,8,11
        '''
        # image_path = Resource.get_path('example_pics/sample11.dataset.jpg')
        # image_path = Resource.get_path('example_pics/sample08.dataset.jpg')
        image_path = Resource.get_path('example_pics/sample16.dataset.jpg')
        # image_path = Resource.get_path('for_issues/cannot_recognize_02.jpg')
        main_sudoku.answer_quiz_with_pic(image_path).pl()
        # gray_image = cv2.imread(image_path, 0)
        # Display.image(gray_image)
