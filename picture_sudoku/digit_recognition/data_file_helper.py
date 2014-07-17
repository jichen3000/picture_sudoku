'''
    for handlering the generate data matrix
    from number ragion files.
'''

import numpy
import os
from picture_sudoku.helpers.common import Resource, OtherResource

IMG_SIZE = 32
FULL_SIZE = 1024


FONT_TRAINING_PATH = OtherResource.get_path('font_training')
FONT_RESULT_PATH = OtherResource.get_path('font_training_result')

SUPPLEMENT_RESULT_PATH = OtherResource.get_path('font_supplement_result')

HAND_TRAINING_PATH = OtherResource.get_path('hand_training')
HAND_TESTING_PATH = OtherResource.get_path('hand_testing')
HAND_RESULT_PATH = Resource.get_path('hand_training_result')

SUPPLEMENT_TRAINING_PATH = OtherResource.get_path('supplement_training')

def get_dataset_matrix_hash(the_path, start_with_numbers):
    '''
        main method
    '''
    return {i:numpy.mat(get_dataset_from_filenames(the_path, 
        filter_filenames_with_nums(the_path,(i,)))) for i in start_with_numbers}

def get_dataset_matrix_hash_with_supplement(
        the_path, supplement_path, start_with_numbers):
    '''
        main method
    '''
    def get_all_dataset(index):
        original_data_matrix = numpy.mat(get_dataset_from_filenames(the_path, 
            filter_filenames_with_nums(the_path,(index,))))
        supplement_filenames = filter_filenames_with_nums(supplement_path,(index,))
        if(len(supplement_filenames) == 0):
            return original_data_matrix
        supplement_data_matrix = numpy.mat(get_dataset_from_filenames(supplement_path, 
            supplement_filenames))
        data_matrix = numpy.concatenate(
            (original_data_matrix, supplement_data_matrix), axis=0)
        return data_matrix
    return {i:get_all_dataset(i) for i in start_with_numbers}

def load_data(the_path, start_with_numbers):
    ''' for test. It loads all number ragion file and 
        return the data matrix, and label matrix.
        One ragion file is gereated to one row of matrix, 
        and one row is the array of binary.
    '''
    file_names = filter_filenames_with_nums(the_path, start_with_numbers)
    data_matrix = numpy.mat(get_dataset_from_filenames(the_path, 
            file_names))
    label_matrix = numpy.mat(get_labels_from_filenames( 
            file_names)).transpose()
    return data_matrix, label_matrix

def load_data_with_supplement(
        the_path, supplement_path, start_with_numbers):
    '''
        for test. It generates data matrix, and label matrix,
        with some supplement number files.
    '''
    original_data_matrix, original_label_matrix = load_data(
        the_path, start_with_numbers)
    supplement_data_matrix, supplement_label_matrix = load_data(
        supplement_path, start_with_numbers)
    if supplement_label_matrix.shape[0] == 0:
        return original_data_matrix, original_label_matrix
    data_matrix = numpy.concatenate(
        (original_data_matrix, supplement_data_matrix), axis=0)
    label_matrix = numpy.concatenate(
        (original_label_matrix, supplement_label_matrix), axis=0)
    return data_matrix, label_matrix

def filter_filenames_with_nums(pathname,start_with_numbers):
    num_strs = map(str, start_with_numbers)
    # num_str = str(start_with_number)
    return [filename for filename in os.listdir(pathname) 
        for num_str in num_strs if filename.startswith(num_str)]

def save_list(file_path, the_list):
    with open(file_path, 'w') as the_file:
        for item in the_list:
            the_file.write("%s\n" % item)


def binary_number_to_lists(file_path):
    with open(file_path) as data_file:
        result = [int(line[index]) for line in data_file 
            for index in range(IMG_SIZE)]
    return result

def binary_number_to_intn_lists(file_path, split_number=16):
    with open(file_path) as data_file:
        result = [int(line[index*split_number:(index+1)*split_number],2) 
            for line in data_file for index in range(IMG_SIZE/split_number)]
    return result

def get_dataset_from_filenames(path_name, file_names, 
    binary_func=binary_number_to_lists):
    return [binary_func(os.path.join(path_name,file_name))
        for file_name in file_names]

def get_label_from_filename(filename):
    return int(filename.split('_')[0])

def get_labels_from_filenames(file_names):
    return map(get_label_from_filename, file_names)





if __name__ == '__main__':
    from minitest import *

    number_file_count = 54

    with test(filter_filenames_with_nums):
        filenames = filter_filenames_with_nums(FONT_TRAINING_PATH, range(10))
        filenames.size().must_equal(number_file_count * 10)
        filenames[0].must_equal('0_normal_bold_antiqua.dataset')
        filenames[-1].must_equal('9_normal_normal_verdana.dataset')

    with test(load_data):
        the_number = 9
        data_matrix, label_matrix = load_data(
                FONT_TRAINING_PATH, [the_number])
        data_matrix.shape.must_equal((number_file_count, FULL_SIZE))
        label_matrix.shape.must_equal((number_file_count, 1))
        label_matrix[0,0].must_equal(the_number)
        label_matrix[-1,0].must_equal(the_number)

    with test(load_data_with_supplement):
        the_number = 6
        filenames = filter_filenames_with_nums(
            SUPPLEMENT_TRAINING_PATH, [the_number])
        supplement_file_count = filenames.size()

        data_matrix, label_matrix = load_data_with_supplement(
            FONT_TRAINING_PATH, SUPPLEMENT_TRAINING_PATH, [the_number])
        data_matrix.shape.must_equal(
            (number_file_count+supplement_file_count, FULL_SIZE))
        label_matrix.shape.must_equal(
            (number_file_count+supplement_file_count, 1))


        the_number = 5
        filenames = filter_filenames_with_nums(
            SUPPLEMENT_TRAINING_PATH, [the_number])
        supplement_file_count = filenames.size()

        data_matrix, label_matrix = load_data_with_supplement(
            FONT_TRAINING_PATH, SUPPLEMENT_TRAINING_PATH, [the_number])
        data_matrix.shape.must_equal(
            (number_file_count+supplement_file_count, FULL_SIZE))
        label_matrix.shape.must_equal(
            (number_file_count+supplement_file_count, 1))
        pass

    with test(get_dataset_matrix_hash):
        the_number = 9
        datase_dict = get_dataset_matrix_hash(FONT_TRAINING_PATH, range(10))
        datase_dict.size().must_equal(10)
        datase_dict[the_number].shape.must_equal((number_file_count, FULL_SIZE))

    with test(get_dataset_matrix_hash_with_supplement):
        the_number = 6
        filenames = filter_filenames_with_nums(
            SUPPLEMENT_TRAINING_PATH, [the_number])
        supplement_file_count = filenames.size()

        datase_dict = get_dataset_matrix_hash_with_supplement(
            FONT_TRAINING_PATH, SUPPLEMENT_TRAINING_PATH, range(10))
        datase_dict.size().must_equal(10)
        datase_dict[the_number].shape.must_equal(
            (number_file_count+supplement_file_count, FULL_SIZE))
