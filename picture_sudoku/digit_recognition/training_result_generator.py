# it will be used to generate the traininig result

import os
import numpy
import cv2

from picture_sudoku.helpers import json_helper
from picture_sudoku.helpers.exceptions import SudokuError
from rbf_smo import Smo
from multiple_svm import MultipleSvm
from multiple_svm import *
import data_file_helper


from picture_sudoku.picture_analyzer import main_analyzer
from picture_sudoku import main_sudoku
from picture_sudoku.cv2_helpers.image import Image
from picture_sudoku.cv2_helpers.ragion import Ragion

IMG_SIZE = data_file_helper.IMG_SIZE
FULL_SIZE = data_file_helper.FULL_SIZE



def extract_real_digit(ragion_file_path):
    prefix = 'real'
    return extract_digit(ragion_file_path, prefix)

def extract_cal_digit(ragion_file_path):
    prefix = 'cal'
    return extract_digit(ragion_file_path, prefix)

def extract_digit(ragion_file_path, prefix):
    basename = os.path.basename(ragion_file_path)
    except_ext_name = os.path.splitext(basename)[0]
    name_parts = except_ext_name.split("_")
    number_names = filter(lambda x: x.startswith(prefix), name_parts)
    if len(number_names) != 1:
        raise SudokuError('The file name {0} do not include the real digit!'
            .format(basename))
    return int(number_names[0][len(prefix):])



def generate_standard_supplement_number_ragions(ragion_file_path, the_digit, 
        save_path=data_file_helper.SUPPLEMENT_TRAINING_PATH): 
    issue_ragion = Image.load_from_txt(ragion_file_path)
    # os.path.basename(ragion_file_path).pl()
    # ragion_file_path.pl()
    resize_interpolation_arr = ((cv2.INTER_AREA, "INTER_AREA"), 
                                (cv2.INTER_LINEAR, "INTER_LINEAR"),
                                (cv2.INTER_CUBIC, "INTER_CUBIC"))
    def resize_and_save_file(interpolation_item):
        interpolation_value, interpolation_name = interpolation_item
        result_file_path = os.path.join(save_path, 
            str(the_digit)+"_"+interpolation_name+"_"+os.path.basename(ragion_file_path))
        resized_ragion = main_sudoku.resize_to_cell_size(issue_ragion, interpolation_value)
        Image.save_to_txt(resized_ragion,result_file_path)
        result_file_path.pl()
        return result_file_path
    return map(resize_and_save_file, resize_interpolation_arr)



if __name__ == '__main__':
    from minitest import *


    def other():
        # dataset_matrix_hash = get_dataset_matrix_hash(training_pic_path, range(2))
        # dataset_matrix_hash = get_dataset_matrix_hash(font_training_path, [0,1])
        # dataset_matrix_hash = get_dataset_matrix_hash(hand_training_path, range(10))
        # dataset_matrix_hash = get_dataset_matrix_hash(training_pic_path, (9,))
        # dataset_matrix_hash.pp()

        # mb = MultipleSvm.load_variables(Smo, 'font_dataset')
        # mb = MultipleSvm.load_variables(Smo, hand_result_path)

        # dataset_matrix_hash[9][0].shape.ppl()
        # mb.dag_classify(dataset_matrix_hash[9][0]).ppl()
        # show_number_matrix(dataset_matrix_hash[9][0])
        # mb.normal_classify(dataset_matrix_hash[9][0]).pp()

        # data_matrix,label_matrix = data_file_helper.load_data(hand_testing_path, range(10))
        # mb.test(data_matrix,label_matrix).pp()
        # training_digits
        # {'error_count': 38, 'error_ratio %': 4.02, 'row_count': 946}
        pass

    def generate_training_result():
        ''' It should be very careful to use this method,
            since it will replace all the result files.
        '''
        arg_exp = 20

        dataset_matrix_hash = data_file_helper.get_dataset_matrix_hash(
            data_file_helper.FONT_TRAINING_PATH, range(1,10))

        mb = MultipleSvm.train_and_save_variables(Smo, dataset_matrix_hash, 
            200, 0.0001, 1000, arg_exp, 
            data_file_helper.SUPPLEMENT_RESULT_PATH)

    def generate_supplement_result():
        ''' It should be very careful to use this method,
            since it will replace all the result files.
        '''
        arg_exp = 20

        dataset_matrix_hash = data_file_helper.get_dataset_matrix_hash_with_supplement(
            data_file_helper.FONT_TRAINING_PATH,
            data_file_helper.SUPPLEMENT_TRAINING_PATH, range(1,10))

        mb = MultipleSvm.train_and_save_variables(Smo, dataset_matrix_hash, 
            200, 0.0001, 1000, arg_exp, 
            data_file_helper.SUPPLEMENT_RESULT_PATH)



    def generate_supplement_number_ragion():
        file_paths=['../../resource/svm_wrong_digits/pic01_no00_real5_cal6.dataset',
                    '../../resource/svm_wrong_digits/pic01_no10_real8_cal6.dataset']
        # file_path = '../../resource/svm_wrong_digits/pic16_no19_real5_cal1.dataset'
        # file_path = '../../resource/svm_wrong_digits/pic17_no13_real6_cal5.dataset'
        # file_path = '../../resource/svm_wrong_digits/pic16_no20_real8_cal6.dataset'
        # file_path = '../../resource/test/pic17_no08_real6_cal8.dataset'
        for file_path in file_paths:
            the_digit = extract_real_digit(file_path)
            generate_standard_supplement_number_ragions(file_path, the_digit,
                    data_file_helper.SUPPLEMENT_TRAINING_PATH)


    def main():
        # generate_supplement_number_ragion()
        # generate_supplement_result()
        pass
    main()

    def test_generate_supplement_number_ragion(
            file_name, simple_svm, supplement_svm):
        file_path = os.path.join(
            data_file_helper.SUPPLEMENT_TRAINING_PATH,file_name)
        the_ragion = Image.load_from_txt(file_path)
        the_number_matrix = numpy.matrix(the_ragion.reshape(1, FULL_SIZE))

        # simple_svm.dag_classify(the_number_matrix).must_equal(
        #     extract_cal_digit(file_path), 
        #     failure_msg="simple file path: {0}".format(file_path))

        supplement_svm.dag_classify(the_number_matrix).must_equal(
            extract_real_digit(file_path), 
            failure_msg="supplement file path: {0}".format(file_path))

    with test(generate_supplement_number_ragion):
        simple_svm = MultipleSvm.load_variables(Smo, 
            data_file_helper.FONT_RESULT_PATH)
        supplement_svm = MultipleSvm.load_variables(Smo, 
            data_file_helper.SUPPLEMENT_RESULT_PATH)

        filenames = data_file_helper.filter_filenames_with_nums(
            data_file_helper.SUPPLEMENT_TRAINING_PATH, range(10))

        for file_name in filenames:
            test_generate_supplement_number_ragion(
                file_name,simple_svm, supplement_svm)


