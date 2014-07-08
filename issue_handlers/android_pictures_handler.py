import cv2
from picture_sudoku.picture_analyzer import main_analyzer
from picture_sudoku.digit_recognition.multiple_svm import MultipleSvm
from picture_sudoku.digit_recognition.rbf_smo import Smo
from picture_sudoku import main_sudoku
from picture_sudoku.cv2_helpers.image import Image


if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.helpers.common import Resource, OtherResource


    # with test("for issue 6 to 8"):
    #     image_path = '../resource/for_issues/cannot_recognize.jpg'
    #     # image_path = '../../resource/for_issues/cannot_recognize_02.jpg'
    #     # number_indexs, number_ragions = main_analyzer.extract_number_ragions(image_path)
    #     # gray_image = cv2.imread(image_path, 0)
    #     # Display.image(gray_image)
    #     issue_ragion = main_analyzer.extract_specified_number_ragion(image_path, 3, 3)
    #     # Display.image_binary(issue_ragion)
    #     # Display.ragions_binary(number_ragions)
    #     # number_indexs.ppl()
    #     # Display.image(numpy_helper.transfer_values_quickly(number_ragions[0],{1:255}))
    #     # number_ragions[0].shape.ppl()
    #     adjusted_ragion =  main_sudoku.adjust_number_ragion(issue_ragion)
    #     transfered_ragion = main_sudoku.transfer_to_digit_matrix(issue_ragion)
    #     # Display.image_binary(adjusted_ragion)
    #     Display.image_binary(main_sudoku.resize_to_cell_size(issue_ragion))
    #     font_result_path = '../other_resource/font_training_result'
    #     mb = MultipleSvm.load_variables(Smo, font_result_path)
    #     mb.dag_classify(transfered_ragion).pl()

    def find_max_contour(threshed_image, filter_func = None, accuracy_percent_with_perimeter=0.0001):
        contours = Image.find_contours(
            threshed_image, filter_func, accuracy_percent_with_perimeter)
        Display.contours(threshed_image, contours)
        if len(contours) == 0:
            return None
        contour_area_arr = [cv2.contourArea(i) for i in contours]
        max_contour = contours[contour_area_arr.index(max(contour_area_arr))]
        return max_contour

    with test("for issue too big"):
        # image_path = Resource.get_path('for_issues/cannot_recognize_02.jpg')
        image_path = Resource.get_path('example_pics/false01.dataset.jpg')
        # image_path = Resource.get_path('example_pics/sample01.dataset.jpg')
        # # # number_indexs, number_ragions = main_analyzer.extract_number_ragions(image_path)
        # gray_image = cv2.imread(image_path, 0)
        # gray_image = cv2.GaussianBlur(gray_image,ksize=(5,5), sigmaX=0)
        
        # threshed_image = cv2.adaptiveThreshold(gray_image, 255,
        #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=7, C=2)

        # # ''' 
        # #     It's very depend on the threshed_image.
        # # '''
        # max_contour = find_max_contour(threshed_image)
        # Display.contours(gray_image, [max_contour])
        main_sudoku.answer_quiz_with_pic(image_path).pl()

        pass
