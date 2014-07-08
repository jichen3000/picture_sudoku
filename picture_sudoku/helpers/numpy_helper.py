import numpy

def transfer_values(arr, rule_hash):
    '''
        rule_hash = {0:1, 255:0}
    '''
    # result = arr.copy()
    result = arr
    for (i, j), value in numpy.ndenumerate(result):
        if value in rule_hash.keys():
            result[i,j] = rule_hash[value]
    return result

def transfer_values_quickly(arr, rule_hash):
    '''
        rule_hash = {0:1, 255:0}
        The feature just like the transfer_values, but it will be quicker.
        In the process of transfer a pic, transfer_values will take about 12 seconds,
        but this one only take 0.16 seconds.
        However, this method cannot handle the rule_hash like {0:1, 1:-1},
        it will transfer all values to the -1.
    '''
    result = arr.copy()
    handled_values = []
    for key, value in rule_hash.items():
        if key in handled_values:
            raise Exception("Cannot support this type of rule_hash %s" %(rule_hash))
        result[result==key]=value
        handled_values.append(value)
    return result

def transfer_1to255(arr):
    return transfer_values_quickly(arr, {1:255})

def transfer_255to1(arr):
    return transfer_values_quickly(arr, {255:1})

def is_array_none(the_array):
    return numpy.all(the_array, None)==None


if __name__ == '__main__':
    from minitest import *
    inject(numpy.allclose, 'must_close')

    with test("transfer_values"):
        arr = numpy.array([[1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1]])
        transfer_values(arr, {0:1, 1:0}).must_equal(
            numpy.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]]), numpy.allclose)

        arr = numpy.array([[1, 1, 1, 2, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1]])
        transfer_values(arr, {1: -1, 2: 1}).must_close(
            numpy.array([[-1, -1, -1,  1, -1, -1, -1],
                         [-1, -1, -1,  1, -1, -1, -1]]))
        arr = numpy.array([[1, 1, 1, 2, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1]])
        transfer_values(arr, {2: 1, 1: -1}).must_close(
            numpy.array([[-1, -1, -1,  1, -1, -1, -1],
                         [-1, -1, -1,  1, -1, -1, -1]]))

        arr = numpy.array([[1, 1, 1, 2, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1]])
        reversed_arr = transfer_values(arr, {2: 1, 1: -1})
        reversed_arr.ppl()
        reversed_transfer_hash = {-1: 1, 1: 2}
        transfer_values(reversed_arr, reversed_transfer_hash).must_close(
            numpy.array([[1, 1, 1, 2, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1]]))

        # notice the issue
        # arr = numpy.array([[1, 1, 1, 2, 1, 1, 1],
        #              [1, 1, 1, 2, 1, 1, 1]])
        # transfer_values_quickly(arr, {1: -1, 2: 1}).must_close(
        #     numpy.array([[-1, -1, -1,  -1, -1, -1, -1],
        #                  [-1, -1, -1,  -1, -1, -1, -1]]))

    with test("transfer_values_quickly"):
        arr = numpy.array([[1, 9],
                     [1, 9]])
        (lambda : transfer_values_quickly(arr, {9:1, 1:-1})).must_raise(Exception, 
            "Cannot support this type of rule_hash {9: 1, 1: -1}")
        transfer_values(arr, {9:1, 1:-1}).must_equal(
            numpy.array([[-1, 1],
                         [-1, 1]]), numpy.allclose)

    with test("is_array_none"):
        is_array_none(None).must_equal(True)
        arr = numpy.array([[1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1]])
        is_array_none(arr).must_equal(False)