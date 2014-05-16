'''
    This file is just used for vertify the function,
    so it cannot be used by others.
'''

import numpy
import cv2

from picture_sudoku.cv2_helpers.display import Display
from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.ragion import Ragion
from picture_sudoku.cv2_helpers.ragion import Ragions


from picture_sudoku.picture_analyzer import main_analyzer
from picture_sudoku.digit_recognition.multiple_svm import MultipleSvm
from picture_sudoku.digit_recognition import multiple_svm
from picture_sudoku.digit_recognition.rbf_smo import Smo
from picture_sudoku.helpers import numpy_helper
from picture_sudoku.helpers import list_helper

from picture_sudoku import main_sudoku

SUDOKU_SIZE = main_analyzer.SUDOKU_SIZE

def show_number_ragions(number_ragions):
    all_number_ragions = Ragions.join_same_size(
            Ragions.fill_to_same_size(number_ragions), 9)
    all_number_ragions = numpy_helper.transfer_values_quickly(all_number_ragions, {1:255})
    Display.image(all_number_ragions)

def gen_pic_test_data():
    result = {  1:
                {'indexs':(0, 1, 4, 9, 12, 13, 14, 19, 20, 25, 27, 31, 35, 36, 39, 41, 44, 45, 49, 53, 55, 60, 61, 66, 67, 68, 71, 76, 79, 80),
                 'digits':(5, 3, 7, 6, 1, 9, 5, 9, 8, 6, 8, 6, 3, 4, 8, 3, 1, 7, 2, 6, 6, 2, 8, 4, 1, 9, 5, 8, 7, 9)},
                2: 
                {'indexs':(1, 2, 4, 10, 11, 12, 13, 17, 20, 21, 26, 27, 29, 32, 33, 34, 37, 39, 43, 44, 46, 47, 49, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 69, 70, 71, 74, 76, 77, 78, 79, 80),
                 'digits':(2, 8, 9, 5, 4, 6, 2, 9, 3, 4, 5, 3, 2, 9, 4, 5, 8, 7, 1, 2, 7, 6, 1, 9, 3, 6, 3, 7, 9, 4, 8, 5, 2, 1, 8, 5, 1, 7, 4, 6, 1, 7, 6, 3, 9, 8)},
                3: 
                {'indexs':(1, 5, 6, 11, 12, 17, 18, 25, 27, 32, 34, 46, 48, 53, 55, 62, 63, 68, 69, 74, 75, 79),
                 'digits':(8, 1, 2, 9, 6, 7, 7, 8, 2, 5, 7, 3, 9, 6, 6, 4, 1, 7, 8, 4, 3, 5)},
                4: 
                {'indexs':(1, 3, 6, 8, 9, 11, 13, 15, 20, 21, 23, 26, 28, 32, 33, 36, 38, 39, 41, 42, 44, 47, 48, 52, 54, 57, 59, 60, 65, 67, 69, 71, 72, 74, 77, 79),
                 'digits':(7, 1, 4, 3, 4, 3, 2, 5, 5, 9, 4, 6, 6, 3, 8, 7, 2, 8, 5, 9, 1, 4, 7, 6, 3, 4, 6, 2, 9, 8, 6, 5, 6, 8, 9, 3)},
                5: 
                {'indexs':(0, 1, 4, 9, 12, 13, 14, 19, 20, 25, 27, 31, 35, 36, 39, 41, 44, 45, 49, 53, 55, 60, 61, 66, 67, 68, 71, 76, 79, 80),
                 'digits':(5, 3, 7, 6, 1, 9, 5, 9, 8, 6, 8, 6, 3, 4, 8, 3, 1, 7, 2, 6, 6, 2, 8, 4, 1, 9, 5, 8, 7, 9)},
                6: 
                {'indexs':(1, 6, 8, 11, 14, 15, 16, 21, 22, 24, 27, 29, 32, 39, 41, 48, 51, 53, 56, 58, 59, 64, 65, 66, 69, 72, 74, 79),
                 'digits':(3, 8, 6, 9, 3, 1, 2, 1, 9, 5, 4, 7, 5, 8, 1, 6, 4, 2, 2, 8, 7, 7, 5, 4, 3, 3, 8, 4)},
                7: 
                # {'indexs':(0, 1, 3, 5, 7, 8, 9, 10, 16, 17, 30, 32, 36, 37, 39, 41, 43, 44, 48, 50, 63, 64, 70, 71, 72, 73, 75, 77, 79, 80),
                #  'digits':(8, 5, 2, 1, 3, 9, 2, 7, 5, 1, 4, 9, 9, 2, 8, 7, 1, 4, 5, 3, 1, 6, 4, 4, 3, 4, 1, 2, 6, 8)},
                {'indexs':(0, 1, 5, 7, 8, 9, 10, 16, 17, 30, 32, 36, 37, 39, 41, 43, 44, 48, 50, 63, 64, 70, 71, 72, 73, 75, 77, 79, 80),
                 'digits':(8, 5, 1, 3, 9, 2, 7, 5, 1, 4, 9, 9, 2, 8, 7, 1, 4, 5, 3, 1, 6, 4, 4, 3, 4, 1, 2, 6, 8)},
                8: 
                {'indexs':(5, 11, 12, 13, 16, 20, 25, 26, 27, 28, 33, 34, 40, 46, 47, 52, 53, 54, 55, 60, 64, 67, 68, 69, 75),
                 'digits':(5, 1, 4, 6, 8, 4, 1, 2, 4, 2, 7, 5, 3, 6, 5, 4, 1, 2, 1, 3, 9, 5, 7, 6, 2)},
                9: 
                {'indexs':(1, 5, 13, 15, 17, 20, 23, 28, 30, 33, 35, 37, 43, 45, 47, 50, 52, 57, 60, 63, 65, 67, 75, 79),
                 'digits':(1, 2, 5, 2, 6, 9, 7, 2, 5, 4, 7, 9, 6, 5, 4, 1, 9, 6, 1, 3, 7, 1, 8, 4)},
                10: 
                {'indexs':(1, 4, 7, 9, 11, 14, 17, 23, 25, 28, 29, 30, 32, 36, 44, 48, 50, 51, 52, 55, 57, 63, 66, 69, 71, 73, 76, 79),
                 'digits':(4, 7, 5, 8, 2, 4, 3, 3, 8, 2, 6, 3, 8, 1, 7, 1, 7, 3, 6, 9, 4, 2, 7, 6, 5, 1, 5, 7)},
                11: 
                {'indexs':(0, 3, 8, 9, 13, 14, 15, 19, 21, 22, 28, 34, 37, 38, 42, 43, 46, 52, 58, 59, 61, 65, 66, 67, 71, 72, 77, 80),
                 'digits':(3, 8, 6, 4, 6, 7, 3, 1, 9, 4, 4, 3, 2, 8, 7, 1, 5, 4, 5, 8, 6, 2, 3, 9, 5, 5, 1, 4)},
                12: 
                {'indexs':(1, 6, 7, 9, 11, 14, 20, 23, 25, 28, 30, 33, 35, 36, 40, 44, 45, 47, 50, 52, 55, 57, 60, 66, 69, 71, 73, 74, 79),
                 'digits':(4, 1, 9, 9, 1, 3, 7, 4, 2, 8, 1, 3, 9, 7, 4, 1, 3, 6, 8, 4, 7, 6, 9, 4, 8, 2, 2, 5, 6)},
                13: 
                {'indexs':(1, 3, 4, 13, 14, 17, 20, 24, 28, 31, 35, 36, 37, 39, 41, 43, 44, 45, 49, 52, 56, 60, 63, 66, 67, 76, 77, 79),
                 'digits':(6, 8, 5, 6, 7, 3, 9, 4, 8, 1, 2, 2, 3, 7, 6, 1, 8, 9, 4, 6, 8, 6, 7, 9, 3, 8, 4, 2)},
                14: 
                {'indexs':(1, 6, 7, 9, 12, 14, 19, 25, 26, 27, 29, 31, 35, 39, 40, 41, 45, 49, 51, 53, 54, 55, 61, 66, 68, 71, 73, 74, 79),
                 'digits':(1, 8, 6, 5, 3, 8, 8, 5, 9, 1, 4, 7, 5, 2, 5, 1, 6, 4, 1, 8, 7, 4, 8, 5, 7, 2, 6, 2, 7)},
                15: 
                {'indexs':(0, 8, 10, 12, 16, 20, 24, 28, 30, 32, 40, 48, 50, 52, 56, 60, 64, 68, 70, 72, 80),
                 'digits':(9, 5, 4, 7, 3, 8, 1, 3, 6, 7, 8, 3, 9, 6, 1, 9, 2, 6, 4, 5, 8)}
            }
    return result


# def transfer_str(the_str):
#     digit_str = the_str.replace(',','')
#     digit_str = digit_str.replace(' ','')

#     return tuple(int(i) for i in digit_str)


# with test(transfer_str):
#     transfer_str('812, 967, 78, 257, , 396, 64, 178, 435').pl()



def print_and_get_difference(actual, expected, image_index):
    print 'vertify no: %d' % (image_index) 
    actual_number_indexs, actual_digits, actual_number_ragions = actual
    expected_number_indexs, expected_digits = expected['indexs'], expected['digits']

    if actual_number_indexs == expected_number_indexs:
        print 'number_indexs are same!'
    else:
        print 'number_indexs are different!'
        print '  actual:'+str(actual_number_indexs)
        print 'expected:'+str(expected_number_indexs)
    if actual_digits == expected_digits:
        print 'digits are same!'
    else:
        print 'digits are different!'
        print '  actual:'+str(actual_digits)
        print 'expected:'+str(expected_digits)
        difference = list_helper.cal_difference(expected_digits, actual_digits)
        print 'difference:'+str(difference)
        save_different_number(image_index, difference, actual_number_ragions)
        return difference
    return False

def save_different_number(image_index, difference, number_ragions):
    def gen_file_path(digit_index, real_digit, cal_digit):
        return '../resource/svm_wrong_digits/pic' + str(image_index).zfill(2) + \
            '_no'+str(digit_index).zfill(2)+'_real'+str(real_digit)+'_cal'+str(cal_digit) + '.dataset'
    for digit_index, real_digit, cal_digit in difference:
        file_path = gen_file_path(digit_index, real_digit, cal_digit)
        Image.save_to_txt(number_ragions[digit_index], file_path)
        print 'save file: '+ file_path


def save_dataset(gray_image, file_path):
    transfered_image = numpy_helper.transfer_values_quickly(gray_image,{255:1})
    return Image.save_to_txt(transfered_image,file_path)


def join_number_ragions(number_indexs, number_ragions):
    all_number_ragions = []
    for index, number_ragion in zip(number_indexs, number_ragions):
        ragion_index = len(all_number_ragions)
        if ragion_index < index:
            all_number_ragions += [numpy.zeros((1,1)) for i in range(index-ragion_index)]            
        all_number_ragions.append(number_ragion)
    all_number_ragions = Ragions.join_same_size(
        Ragions.fill_to_same_size(all_number_ragions), 9)
    return all_number_ragions

def identify_wrong_number(number_ragions, difference, show_actual=True):
    all_number_images = generate_number_images()
    identified_number_ragions = number_ragions[:]
    for index,_,actual in difference:
        height, width = number_ragions[index].shape
        if show_actual:
            number_ragions[index] = Image.resize_keeping_ratio_by_height(all_number_images[actual], height)
            height, width = number_ragions[index].shape
        identified_number_ragions[index] = Ragion.fill(number_ragions[index], (height+2, width+2), 1)
    return identified_number_ragions

def generate_wrong_number_ragions(difference):
    all_number_images = generate_number_images()
    all_number_ragions = [numpy.zeros((1,1)) for i in range(SUDOKU_SIZE*SUDOKU_SIZE)]
    for index, expected,actual in difference:
        all_number_ragions[index] = all_number_images[actual]
    return all_number_ragions

def generate_number_images():
    font_training_path = '../other_resource/font_training'
    suffix_file_name = '_normal_normal_garamond.dataset'
    def generate_number(index):
        file_path = font_training_path+"/"+str(index)+suffix_file_name
        return Image.read_from_number_file(file_path)
    return map(generate_number, range(10))



def show_difference(pic_file_path, actual, difference):
    the_image = cv2.imread(pic_file_path, 0)
    the_image = Image.resize_keeping_ratio_by_height(the_image)
    # Display.image(the_image)

    actual_number_indexs, actual_digits, actual_number_ragions = actual
    if difference:
        identified_number_ragions = identify_wrong_number(actual_number_ragions, difference, False)
        # identified_number_ragions = identify_wrong_number(actual_number_ragions, difference)
    else:
        identified_number_ragions = actual_number_ragions
    all_number_ragion = join_number_ragions(actual_number_indexs, identified_number_ragions)
    all_number_ragion = numpy_helper.transfer_values_quickly(all_number_ragion,{1:255})
    # all_number_ragion = Image.colorize(all_number_ragion)
    # Display.image(all_number_ragion)

    if difference:
        wrong_number_ragions = generate_wrong_number_ragions(difference)
        wrong_number_ragion = join_number_ragions(actual_number_indexs, wrong_number_ragions)
        wrong_number_ragion = Image.resize_keeping_ratio_by_height(wrong_number_ragion, all_number_ragion.shape[0])
        wrong_number_ragion = numpy_helper.transfer_values_quickly(wrong_number_ragion,{1:255})

        # Display.image(Ragions.join((all_number_ragion, wrong_number_ragion), 1))
        # Display.ragions([the_image, all_number_ragion, wrong_number_ragion])
        Display.images([Ragions.join((all_number_ragion, wrong_number_ragion),1), the_image])
    else:
        Display.images([all_number_ragion, the_image])


FONT_RESULT_PATH = '../resource/digit_recognition/font_training_result'

def digit_recognize():
    mb = MultipleSvm.load_variables(Smo, FONT_RESULT_PATH)

    # file_path = '../resource/svm_wrong_digits/pic04_no17_real8_cal3.dataset'
    file_path = '../resource/svm_wrong_digits/pic04_no33_real8_cal3.dataset'
    # file_path = '../resource/svm_wrong_digits/pic15_no19_real5_cal6_1.dataset'
    # file_path = '../resource/svm_wrong_digits/pic15_no19_real5_cal6.dataset'
    number_ragion = numpy.mat(Image.read_from_number_file(file_path))
    transfered_ragion = numpy_helper.transfer_1to255(number_ragion)
    # adjusted_ragion = main_sudoku.adjust_number_ragion(transfered_ragion)
    adjusted_ragion = adjust_number_ragion2(transfered_ragion)
    # adjusted_ragion = transfered_ragion
    Display.ragions([transfered_ragion, adjusted_ragion])
    adjusted_ragion = numpy_helper.transfer_255to1(adjusted_ragion)
    number_matrix = main_sudoku.transfer_to_digit_matrix(adjusted_ragion)
    mb.dag_classify(number_matrix).ppl()

def adjust_number_ragion2(the_image):
    # element_value: {0: 'Rect', 1: 'Cross', 2: 'Ellipse'}
    element_value = 1
    kernel_size_value = 2*1+1
    kernel = cv2.getStructuringElement(element_value, (kernel_size_value, kernel_size_value))

    # return the_image
    # return cv2.erode(the_image, kernel)
    # opening
    # return cv2.morphologyEx(the_image, 2, kernel)
    # closing = erode(dilate(the_image, kernel))    
    return cv2.morphologyEx(the_image, 3, kernel)
    # return cv2.morphologyEx(the_image, 5, kernel)

    # return cv2.dilate(the_image, kernel)



def vertify_all_pics():
    pic_data = gen_pic_test_data()

    # hand_result_path = '../resource/digit_recognition/hand_dataset'
    smo_svm = MultipleSvm.load_variables(Smo, FONT_RESULT_PATH)

    def handle_one(i, extend_name='jpg'):
        pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.'+extend_name
        actual = main_sudoku.get_digits(pic_file_path, smo_svm, True)
        difference = print_and_get_difference(actual, pic_data[i], i)
        show_difference(pic_file_path, actual, difference)
        return True

    # handle_one(7)
    # handle_one(4)
    handle_one(15, 'png')
    map(handle_one, range(1,15))


if __name__ == '__main__':
    from minitest import *
    '''
        README: 
            Using digit_recognize, you can vertify a single number image with svm.
            Using vertify_all_pics, you can vertify all numbers in a image with svm,
                it will show the difference, and save the wrong number image.
    '''
    def main():
        # digit_recognize()
        vertify_all_pics()
        pass
        # 4 8-3

    with test(main):
        main()