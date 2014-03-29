import numpy

def transfer_values(arr, rule_hash):
    '''
        rule_hash = {0:1, 255:0}
    '''
    # result = arr.copy()
    result = arr
    for (i, j), value in numpy.ndenumerate(result):
        # if value in rule_hash.keys():
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


if __name__ == '__main__':
    from minitest import *

    with test("transfer_values"):
        arr = numpy.array([[1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1]])
        transfer_values(arr, {0:1, 1:0}).must_equal(
            numpy.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]]), numpy.allclose)

    with test("transfer_values_quickly"):
        # arr = numpy.array([[1, 1, 1, -1, 1, 1, 1],
        #              [1, 1, 1, -1, 1, 1, 1]])
        # transfer_values_quickly(arr, {1:0, -1:2}).must_equal(
        #     numpy.array([[0, 0, 0, 2, 0, 0, 0],
        #                  [0, 0, 0, 2, 0, 0, 0]]), numpy.allclose)

        arr = numpy.array([[1, 9],
                     [1, 9]])
        (lambda : transfer_values_quickly(arr, {9:1, 1:-1})).must_raise(Exception, 
            "Cannot support this type of rule_hash {9: 1, 1: -1}")
        transfer_values(arr, {9:1, 1:-1}).must_equal(
            numpy.array([[-1, 1],
                         [-1, 1]]), numpy.allclose)