'''
    this version copied from the book, and I've tested it with the data,
    the result is perfect.
'''

import os
import numpy

from picture_sudoku.helpers import json_helper
from rbf_smo import Smo

IMG_SIZE = 32



def most_common(lst):
    return max(set(lst), key=lst.count)

def is_any_member_in_list(source_list, target_list):
    is_in_list = tuple((label in target_list) for label in source_list)
    return any(is_in_list)

def flatten_for_two_layers(the_list):
    return [item for sub_list in the_list for item in sub_list]

        


''' split line '''
''' for multiple binary'''
def get_all_posible_indexs(the_list):
    return tuple((i,j) for i in the_list for j in the_list if i<j)

class MultipleSvm(object):
    '''
        one against one
        Direct Acyclic Graph, DAG will let classify more efficient.
    '''
    def __init__(self, binary_class):
        self.binary_class = binary_class
        self.classifying_hash = {}
        
    variables_file_name = 'multiple_binary.finaldata'

    def build_classifying_objects(self, data_matrix_hash, 
                edge_threshold, tolerance, max_iteration_count, arg_exp, data_path):
        def gen_label_matrix(label_value, count):
            return numpy.mat(numpy.zeros((count, 1), numpy.int8)) + label_value

        for label_i, label_j in get_all_posible_indexs(data_matrix_hash.keys()):
            data_matrix = numpy.concatenate((data_matrix_hash[label_i], 
                                       data_matrix_hash[label_j]), axis=0)
            label_i_matrix = gen_label_matrix(label_i, data_matrix_hash[label_i].shape[0])
            label_j_matrix = gen_label_matrix(label_j, data_matrix_hash[label_j].shape[0])
            label_matrix = numpy.concatenate((label_i_matrix, label_j_matrix), axis=0)

            classifying_key = (label_i, label_j)
            file_prefix = '_'.join((str(label_i),  str(label_j), ''))
            # transfer_hash = {label_i:-1, label_j:1}
            transfer_hash = {label_j:-1, label_i:1}

            self.label_tuple = tuple(data_matrix_hash.keys())

            smo = self.binary_class.train(data_matrix,label_matrix, 
                    edge_threshold, tolerance, max_iteration_count, arg_exp, transfer_hash)
            # smo.transfer_hash.ppl()
            # smo.reversed_transfer_hash.ppl()
            smo.test(data_matrix,label_matrix).pp()
            smo.save_variables(data_path, prefix = file_prefix)
            self.classifying_hash[classifying_key]=smo

        return self

    def normal_classify(self, row_matrix, with_order_labes=False):
        '''
            It's the normal classify, it will take more times ((len(labels)-1)/2) than the dag, 
            but the answer almost equal with dag.
            {'error_count': 36, 'error_ratio %': 3.81, 'row_count': 946}
            Finished tests in 94.411094s.
        '''
        occurence_hash = { classifying_key:classifying_object.classify(row_matrix)
                for classifying_key, classifying_object in self.classifying_hash.items()}
        if with_order_labes:
            return most_common(occurence_hash.values()), occurence_hash
        return most_common(occurence_hash.values())


    def dag_classify(self, row_matrix, with_order_labes=False):
        '''
            Direct Acyclic Graph, DAG will let classify more efficient.
            The classify times are only the length of labels.

            {'error_count': 37, 'error_ratio %': 3.91, 'row_count': 946}
            Finished tests in 20.857392s.    
        '''
        predict_label = self.label_tuple[0]
        order_labels = []
        for next_label in self.label_tuple[1:]:
            classifying_key = (predict_label, next_label)
            predict_label = self.classifying_hash[classifying_key].classify(row_matrix)
            order_labels.append([(predict_label,)]+list(classifying_key))

        if with_order_labes:
            return predict_label, order_labels
        return predict_label

    def __package_info(self, error_count, row_count):
        return {'error_ratio %':round(float(error_count)/row_count * 100, 2),
                'error_count': error_count,
                'row_count': row_count}

    def test(self, testing_data_matrix, testing_label_matrix):
        row_count = testing_data_matrix.shape[0]

        # errors = tuple(self.classify(testing_data_matrix[i, :])[0]!=testing_label_matrix[i,0] 
        #     for i in range(row_count))
        # self.last_test_info = self.__package_info(errors.count(True), row_count)
        # return self.last_test_info

        error_count = 0
        result_list = []
        for i in range(row_count):
            result = self.dag_classify(testing_data_matrix[i, :])
            if result[0]!=testing_label_matrix[i,0]:
                error_count += 1
                result_list.append([testing_label_matrix[i,0]]+list(result))

            # if error_count > 40:
            #     break
        # result_list.ppl()

        self.last_test_info = self.__package_info(error_count, row_count)
        return self.last_test_info

    def save_variables(self, data_path):
        file_path = os.path.join(data_path, self.variables_file_name)
        json_helper.save_nicely(file_path, self.classifying_hash.keys())
        return True

    def gen_prefix_from_list(self, the_list):
        ''' 
            the_list like: (0, 9)
            result: '_0_9_'
        '''
        return '_'.join(map(str, the_list) + [''])

    def gen_binary_transfer_hash(self, the_list):
        ''' 
            the_list like: (0, 9)
            result: {0: -1, 9:1}
        '''
        return {the_list[0]:-1, the_list[1]:1}

    @classmethod
    def load_variables(cls, binary_class, data_path):
        self = MultipleSvm(binary_class)
        # dir(self).pp()
        file_path = os.path.join(data_path, self.variables_file_name)
        keys = json_helper.load(file_path)
        keys = map(tuple, keys)
        self.label_tuple = tuple(set(flatten_for_two_layers(keys)))

        for classifying_key in keys:
            file_prefix = self.gen_prefix_from_list(classifying_key)
            transfer_hash = self.gen_binary_transfer_hash(classifying_key)

            self.classifying_hash[classifying_key] = self.binary_class.load_variables(
                data_path, file_prefix)

        return self


    @classmethod
    def train_and_save_variables(cls, binary_class, data_matrix_hash,
            edge_threshold, tolerance, max_iteration_count, arg_exp, data_path='dataset'):

        this = cls(binary_class)
        this.build_classifying_objects(data_matrix_hash,
            edge_threshold, tolerance, max_iteration_count, arg_exp, data_path)
        this.save_variables(data_path)
        return this


''' split line '''
''' for digit recongnition'''

def filter_filenames_with_nums(pathname,start_with_numbers):
    num_strs = map(str, start_with_numbers)
    # num_str = str(start_with_number)
    return [filename for filename in os.listdir(pathname) 
        for num_str in num_strs if filename.startswith(num_str)]

def save_list(file_path, the_list):
    with open(file_path, 'w') as the_file:
        for item in the_list:
            the_file.write("%s\n" % item)

def load_data_from_images_with_nums(the_path, start_with_numbers):
    file_names = filter_filenames_with_nums(the_path, start_with_numbers)
    data_matrix = numpy.mat(get_dataset_from_filenames(the_path, 
            file_names))
    label_matrix = numpy.mat(get_labels_from_filenames( 
            file_names)).transpose()
    return data_matrix, label_matrix


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



def get_dataset_matrix_hash(the_path, start_with_numbers):
    return {i:numpy.mat(get_dataset_from_filenames(the_path, 
        filter_filenames_with_nums(the_path,(i,)))) for i in start_with_numbers}



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

    def show_number_matrix(number_matrix):
        from picture_sudoku.helpers import numpy_helper
        from picture_sudoku.cv2_helpers.display import Display
        from picture_sudoku.cv2_helpers.image import Image
        binary_number_image = number_matrix.reshape((IMG_SIZE, IMG_SIZE))
        number_image = numpy_helper.transfer_values_quickly(binary_number_image, {1:255})
        number_image = numpy.array(number_image, dtype=numpy.uint8)
        # Image.save_to_txt(number_image,'test1.dataset')
        Display.image(number_image)

    def test_multi():
        arg_exp = 20

        # dataset_matrix_hash = get_dataset_matrix_hash(training_pic_path, range(2))
        # dataset_matrix_hash = get_dataset_matrix_hash(font_training_path, range(1,10))
        # dataset_matrix_hash = get_dataset_matrix_hash(font_training_path, [0,1])
        # dataset_matrix_hash = get_dataset_matrix_hash(hand_training_path, range(10))
        # dataset_matrix_hash = get_dataset_matrix_hash(training_pic_path, (9,))
        # dataset_matrix_hash.pp()

        # mb = MultipleSvm.train_and_save_variables(Smo, dataset_matrix_hash, 200, 0.0001, 1000, arg_exp, font_result_path)
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
