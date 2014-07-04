# it will be used to generate the traininig result

import os
import numpy

from picture_sudoku.helpers import json_helper
from rbf_smo import Smo
from multiple_svm import MultipleSvm
from multiple_svm import *


if __name__ == '__main__':
    from minitest import *

    font_result_path = '../../other_resource/font_training_result'
    hand_result_path = '../../other_resource/hand_training_result'

    # font_training_path = '../../../codes/python/projects/font_number_binary/number_images'
    font_training_path = '../../other_resource/font_training'

    # hand_training_path = '../../../codes/python/ml/k_nearest_neighbours/training_digits'
    # hand_testing_path = '../../../codes/python/ml/k_nearest_neighbours/test_digits'
    hand_training_path = '../../other_resource/hand_training'
    hand_testing_path = '../../other_resource/hand_testing'


    def test_multi():
        arg_exp = 20

        # dataset_matrix_hash = get_dataset_matrix_hash(training_pic_path, range(2))
        dataset_matrix_hash = get_dataset_matrix_hash(font_training_path, range(1,10))
        # dataset_matrix_hash = get_dataset_matrix_hash(font_training_path, [0,1])
        # dataset_matrix_hash = get_dataset_matrix_hash(hand_training_path, range(10))
        # dataset_matrix_hash = get_dataset_matrix_hash(training_pic_path, (9,))
        # dataset_matrix_hash.pp()

        mb = MultipleSvm.train_and_save_variables(Smo, dataset_matrix_hash, 200, 0.0001, 1000, arg_exp, font_result_path)
        # mb = MultipleSvm.load_variables(Smo, 'font_dataset')
        # mb = MultipleSvm.load_variables(Smo, hand_result_path)

        # dataset_matrix_hash[9][0].shape.ppl()
        # mb.dag_classify(dataset_matrix_hash[9][0]).ppl()
        # show_number_matrix(dataset_matrix_hash[9][0])
        # mb.normal_classify(dataset_matrix_hash[9][0]).pp()

        # data_matrix,label_matrix = load_data_from_images_with_nums(hand_testing_path, range(10))
        # mb.test(data_matrix,label_matrix).pp()
        # training_digits
        # {'error_count': 38, 'error_ratio %': 4.02, 'row_count': 946}
        pass


    with test("test_multi"):
        test_multi()
        pass
