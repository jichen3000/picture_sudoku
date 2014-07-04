'''
    sequential minimal optimization (SMO)
    John C. Platt, "Using Analytic QP and Sparseness 
    to Speed Training of Support Vector Machines"

    radial bias function,
    mapping from one feature space to another feature space.
    inner products.
    One great thing about the SVM optimization is that all 
    operations can be written in terms of inner products. 
    Inner products are two vectors multiplied together to 
    yield a scalar or single number.

    kernel trick or kernel substation.
    A popular kernel is the radial bias function, which we'll introduce next.
'''

import os
import numpy

from picture_sudoku.helpers import json_helper
from picture_sudoku.helpers import numpy_helper



def random_select(except_value, range_max):
    ''' select a value from 0 to m randomly, but not equal the value of except_value '''
    result=except_value
    while (result==except_value):
        result = int(numpy.random.uniform(0,range_max))
    return result



def clip_value(value, high_threshold,low_threshold):
    ''' reset a value according to a range from low_threshold to  high_threshold '''
    if value >  high_threshold:
        value =  high_threshold
    if value < low_threshold:
        value = low_threshold
    return value

def rbf_kernel(data_matrix, row_matrix, arg_exp):
    row_count = data_matrix.shape[0]
    transfered_row_matrix = numpy.mat(numpy.zeros((row_count,1)))
    for row_index in range(row_count):
        delta_row = data_matrix[row_index,:] - row_matrix
        transfered_row_matrix[row_index] = delta_row * delta_row.T
    transfered_row_matrix = numpy.exp( transfered_row_matrix / (-1*arg_exp**2))
    return transfered_row_matrix


class SmoBasic(object):
    def __init__(self, data_matrix, label_matrix, edge_threshold, tolerance, arg_exp):
        self.data_matrix =  data_matrix
        self.label_matrix = label_matrix
        self.edge_threshold = edge_threshold
        self.tolerance = tolerance
        self.row_count, self.col_count = data_matrix.shape
        self.alphas = numpy.mat(numpy.zeros((self.row_count,1)))
        self.b = 0
        self.error_cache = numpy.mat(numpy.zeros((self.row_count,2)))
        self.k_data_matrix = numpy.mat(numpy.zeros((self.row_count,self.row_count)))
        for i in range(self.row_count):
            self.k_data_matrix[:,i] = rbf_kernel(self.data_matrix, self.data_matrix[i,:], arg_exp)


    def calculate_error(self, row_index):
        fx = float(numpy.multiply(self.alphas,self.label_matrix).T * self.k_data_matrix[:,row_index]) + self.b
        return fx - float(self.label_matrix[row_index])

    def get_valid_error_cache_index_list(self):
        ''' get the first col as array, and get the non zero index values list'''
        return numpy.nonzero(self.error_cache[:,0].A)[0]


    def get_index_and_max_delta_error(self, valid_error_cache_index_list, first_alpha_error):
        ''' choose second alpha which has the max delta error with first one'''
        # def cal_error_delta(index):
        #     delta_error = abs(first_alpha_error - self.calculate_error(index))
        #     return delta_error
        # valid_error_cache_index_list.pp()
        # error_delta_list = [abs(first_alpha_error - self.calculate_error(i)) for i in valid_error_cache_index_list]
        # # error_delta_list = [(-1,matrix([[ 0.0]]))] + error_delta_list
        # # error_delta_list.pp()
        # # first_alpha_error.pp()
        # # self.calculate_error(5).pp()
        # index, value = get_index_and_max_value(error_delta_list)
        # if not (value > 0.0) :
        #     index = -1
        # (index, value).pp()
        # return index, value
        second_alpha_index = -1
        max_delta_error = 0
        second_alpha_error = 0
        for k in valid_error_cache_index_list:
            error_k = self.calculate_error(k)
            delta_error = abs(first_alpha_error - error_k)
            if (delta_error > max_delta_error):
                second_alpha_index = k
                max_delta_error = delta_error
                second_alpha_error = error_k
        # (second_alpha_index, second_alpha_error).pp()
        return second_alpha_index, second_alpha_error

    def is_error_in_tolerance(self, error_value, index):
        return ((self.label_matrix[index]*error_value < -self.tolerance) 
            and (self.alphas[index] < self.edge_threshold)) or \
            ((self.label_matrix[index]*error_value > self.tolerance) and (self.alphas[index] > 0))

    def cal_edegs(self, first_alpha_index, second_alpha_index):
        if (self.label_matrix[first_alpha_index] != self.label_matrix[second_alpha_index]):
            L = max(0, self.alphas[second_alpha_index] - self.alphas[first_alpha_index])
            H = min(self.edge_threshold, self.edge_threshold + self.alphas[second_alpha_index] - 
                self.alphas[first_alpha_index])
        else:
            L = max(0, self.alphas[second_alpha_index] + self.alphas[first_alpha_index] - self.edge_threshold)
            H = min(self.edge_threshold, self.alphas[second_alpha_index] + self.alphas[first_alpha_index])
        return L,H

    def cal_eta(self, first_alpha_index, second_alpha_index):
        # return 2.0 * self.data_matrix[first_alpha_index,:]*self.data_matrix[second_alpha_index,:].T - \
        #     self.data_matrix[first_alpha_index,:]*self.data_matrix[first_alpha_index,:].T - \
        #     self.data_matrix[second_alpha_index,:]*self.data_matrix[second_alpha_index,:].T
        return 2.0 * self.k_data_matrix[first_alpha_index,second_alpha_index] - \
            self.k_data_matrix[first_alpha_index,first_alpha_index] - self.k_data_matrix[second_alpha_index,second_alpha_index]


    def cal_second_alpha(self, first_alpha_error, 
            second_alpha_index, second_alpha_error, second_alpha_value, eta, H, L):
        result = second_alpha_value - self.label_matrix[second_alpha_index]* \
            (first_alpha_error - second_alpha_error)/eta
        result = clip_value(result,H,L)
        return result, (result-second_alpha_value)
        # self.alphas[second_alpha_index] -= self.label_matrix[second_alpha_index]* \
        #     (first_alpha_error - second_alpha_error)/eta
        # self.alphas[second_alpha_index] = clip_value(self.alphas[second_alpha_index],H,L)

    def is_moving_enough(self, delta_value):
        return (abs(delta_value) >= 0.00001)

    def cal_first_alpha(self, first_alpha_index, first_alpha_value, 
            second_alpha_index, delta_second_alpha):
        result = first_alpha_value - self.label_matrix[second_alpha_index] * \
            self.label_matrix[first_alpha_index] * delta_second_alpha
        return result, (result - first_alpha_value)

    def cal_b(self, b, first_alpha_index, first_alpha_error, delta_first_alpha,
            second_alpha_index, second_alpha_error, delta_second_alpha):

        delta_first_label = self.label_matrix[first_alpha_index]*(delta_first_alpha)
        delta_second_label = self.label_matrix[second_alpha_index]*(delta_second_alpha)

        b1 = b - first_alpha_error - \
                delta_first_label * self.k_data_matrix[first_alpha_index,first_alpha_index] - \
                delta_second_label * self.k_data_matrix[first_alpha_index,second_alpha_index] 
        b1 = b1[0,0]
        if (0 < self.alphas[first_alpha_index]) and (self.edge_threshold > self.alphas[first_alpha_index]): 
            return b1

        b2 = b - second_alpha_error - \
                delta_first_label * self.k_data_matrix[first_alpha_index,second_alpha_index] - \
                delta_second_label * self.k_data_matrix[second_alpha_index,second_alpha_index]
        b2 = b2[0,0]
        if (0 < self.alphas[second_alpha_index]) and (self.edge_threshold > self.alphas[second_alpha_index]): 
            return b2
        
        return (b1 + b2)/2.0


    # main
    def select_second_alpha(self, first_alpha_index, first_alpha_error):
        ''' selects the second alpha, or the inner loop alpha 
            the goal is to choose the second alpha so that 
            we'll take the maximum step during each optimization.'''

        valid_error_cache_index_list = self.get_valid_error_cache_index_list()

        self.update_error_cache(first_alpha_index, first_alpha_error)

        if (len(valid_error_cache_index_list)) > 0:
            second_alpha_index, second_alpha_error = self.get_index_and_max_delta_error(
                valid_error_cache_index_list, first_alpha_error)
        else:
            second_alpha_index = random_select(first_alpha_index, self.row_count)
            second_alpha_error = self.calculate_error(second_alpha_index)

        return second_alpha_index, second_alpha_error

    def update_error_cache(self, index, error_value=None):
        if error_value:
            self.error_cache[index] = [1, error_value]
        else:
            self.error_cache[index] = [1, self.calculate_error(index)]

    # main
    def choose_alphas(self, first_alpha_index):
        first_alpha_error = self.calculate_error(first_alpha_index)
        if not self.is_error_in_tolerance(first_alpha_error, first_alpha_index):
            return 0

        second_alpha_index,second_alpha_error = self.select_second_alpha(first_alpha_index, first_alpha_error)
        L, H = self.cal_edegs(first_alpha_index,second_alpha_index)        
        # Guarantee alphas stay between 0 and edge_threshold
        if L==H: 
            return 0

        eta = self.cal_eta(first_alpha_index, second_alpha_index)
        # Eta is the optimal amount to change alpha[second_alpha_index].
        if eta >= 0: 
            return 0


        self.alphas[second_alpha_index], delta_second_alpha = self.cal_second_alpha(
            first_alpha_error, second_alpha_index, second_alpha_error, self.alphas[second_alpha_index], eta, H, L)

        self.update_error_cache(second_alpha_index)

        if not self.is_moving_enough(delta_second_alpha):
            return 0

        self.alphas[first_alpha_index],  delta_first_alpha = self.cal_first_alpha(
            first_alpha_index, self.alphas[first_alpha_index], 
            second_alpha_index, delta_second_alpha)

        self.update_error_cache(first_alpha_index)

        self.b = self.cal_b(self.b, first_alpha_index, first_alpha_error, delta_first_alpha,
            second_alpha_index, second_alpha_error, delta_second_alpha)
        return 1
        
    def get_non_bound_index_list(self):
        return numpy.nonzero((self.alphas.A > 0) * (self.alphas.A < self.edge_threshold))[0]

    # main
    def cal_alphas_and_b(self, max_iteration_count): 
        # self.init_for_recal_alphas_and_b(arg_exp)
        is_entire_set = True
        alpha_pairs_changed_count = 0
        iter_index = 0
        # you pass through the entire set without changing any alpha pairs.
        while (iter_index < max_iteration_count) and \
                ((alpha_pairs_changed_count > 0) or (is_entire_set)):
            if is_entire_set:
                index_range = range(self.row_count)
            else:
                index_range = self.get_non_bound_index_list()
            alpha_pairs_changed_count = sum(map(self.choose_alphas, index_range))
            if is_entire_set: 
                is_entire_set = False
            elif (alpha_pairs_changed_count == 0): 
                is_entire_set = True
            iter_index += 1
        return self.alphas, self.b

class Smo(object):
    def __init__(self, selected_data_matrix, selected_alphas_label_matrix, b, arg_exp, transfer_hash):
        self.selected_data_matrix = selected_data_matrix
        self.selected_alphas_label_matrix = selected_alphas_label_matrix
        self.svm_count = selected_alphas_label_matrix.shape[0]
        self.b = b
        self.arg_exp = arg_exp
        self.transfer_hash = transfer_hash
        self.reversed_transfer_hash = {v:k for k, v in self.transfer_hash.items()}

        self.last_test_info = {}

    @classmethod
    def train(cls, data_matrix, label_matrix, edge_threshold, 
            tolerance, max_iteration_count, arg_exp, transfer_hash={1:1, -1:-1}):
        label_matrix = numpy_helper.transfer_values(label_matrix, transfer_hash)
        smo_basic = SmoBasic(data_matrix, label_matrix,edge_threshold,tolerance,arg_exp) 
        alphas,b =  smo_basic.cal_alphas_and_b(max_iteration_count)

        nonzero_indexs = numpy.nonzero(alphas.A>0)[0]
        selected_data_matrix = data_matrix[nonzero_indexs]
        selected_alphas_label_matrix = numpy.multiply(label_matrix[nonzero_indexs], alphas[nonzero_indexs])

        return cls(selected_data_matrix, selected_alphas_label_matrix, b, arg_exp, transfer_hash)

    selected_data_matrix_name = 'selected_data_matrix.finaldata'+'.npy'
    selected_alphas_label_matrix_name = 'selected_alphas_label_matrix.finaldata'+'.npy'
    other_variables_name = 'other_variables.finaldata'

    @classmethod
    def get_file_paths(cls, data_path, prefix):
        def gen_path(name):
            return os.path.join(data_path, prefix+name)
        paths ={'selected_data_matrix': gen_path(cls.selected_data_matrix_name),
                'selected_alphas_label_matrix': gen_path(cls.selected_alphas_label_matrix_name),
                'other_variables': gen_path(cls.other_variables_name)
                }
        return paths


    @classmethod
    def load_variables(cls, data_path='dataset', prefix=''):
        paths = cls.get_file_paths(data_path, prefix)

        selected_data_matrix = numpy.load(paths['selected_data_matrix'])
        selected_alphas_label_matrix = numpy.load(paths['selected_alphas_label_matrix'])

        other_variables = json_helper.load(paths['other_variables'])

        other_variables['transfer_hash'] = dict(other_variables['transfer_hash'])

        return cls(selected_data_matrix, selected_alphas_label_matrix, 
                other_variables['b'], other_variables['arg_exp'], other_variables['transfer_hash'])

    def save_variables(self, data_path='dataset', prefix=''):
        paths = self.get_file_paths(data_path, prefix)

        numpy.save(paths['selected_data_matrix'], self.selected_data_matrix)
        numpy.save(paths['selected_alphas_label_matrix'], self.selected_alphas_label_matrix)

        # notice, I save the dict.items(), since json will automatically convert the key to string.
        other_variables = { 'b': self.b, 
                            'arg_exp': self.arg_exp, 
                            'transfer_hash': self.transfer_hash.items(),
                            'svm_count': self.svm_count,
                            'last_test_info': self.last_test_info}
        # other_variables.pp()
        json_helper.save_nicely(paths['other_variables'], other_variables)
        return True

    def __package_info(self, error_count, row_count):
        return {'error_ratio %':round(float(error_count)/row_count * 100, 2),
                'error_count': error_count,
                'row_count': row_count,
                'svm_count': self.svm_count,
                'transfer_hash': self.transfer_hash}

    def test(self, testing_data_matrix, testing_label_matrix):
        testing_label_matrix = numpy_helper.transfer_values(testing_label_matrix, self.transfer_hash)
        row_count = testing_data_matrix.shape[0]

        errors = tuple(self.__classify(testing_data_matrix[i, :])!=numpy.sign(testing_label_matrix[i,0]) 
            for i in range(row_count))
        self.last_test_info = self.__package_info(errors.count(True), row_count)
        return self.last_test_info

    def __classify(self, row_matrix):
        kernel_value = rbf_kernel(self.selected_data_matrix, 
            row_matrix, self.arg_exp)
        predict = kernel_value.T * self.selected_alphas_label_matrix + self.b
        return numpy.sign(predict[0,0])

    def classify(self, row_matrix):
        return self.reversed_transfer_hash[
                self.__classify(row_matrix)]

if __name__ == '__main__':
    from minitest import *

    test_file_path = '../../resource/digit_recognition/test/'
    def gp(file_name):
        return os.path.join(test_file_path, file_name)
    training_path = gp('test_set_RBF.dataset')
    testing_path = gp('test_set_RBF2.dataset')

    def get_dataset_from_file(file_name):
        with open(file_name) as datafile:
            words = [line.strip().split('\t') for line in datafile]
        dataset = [ [float(cell) for cell in row[:-1]] for row in words]
        labels = [float(row[-1]) for row in words]
        return dataset, labels

    def matrix_dataset_labels(data_tuple):
        return numpy.mat(data_tuple[0]), numpy.mat(data_tuple[1]).transpose()

    def get_data_matrix_from_file(file_name):
        return matrix_dataset_labels(get_dataset_from_file(file_name))



    with test("train, save and test"):
        training_data_matrix, training_label_matrix = get_data_matrix_from_file(training_path)
        training_data_matrix.shape.must_equal((100, 2))
        training_label_matrix.shape.must_equal((100, 1))


        smo = Smo.train(training_data_matrix, training_label_matrix, 
            edge_threshold=200, tolerance=0.0001, max_iteration_count=10000, arg_exp=1.3)

        smo.test(training_data_matrix, training_label_matrix).pp()
        smo.save_variables(test_file_path)

    with test("load and test"):
        smo = Smo.load_variables(test_file_path)
        testing_data_matrix, testing_label_matrix = get_data_matrix_from_file(testing_path)
        smo.test(testing_data_matrix, testing_label_matrix).pp()

        smo.classify(testing_data_matrix[0]).must_equal(testing_label_matrix[0,0])


