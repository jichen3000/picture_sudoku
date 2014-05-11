import numpy

def is_in_range(value, range_or_number, percent=0):
    '''
        The range_or_number could be range_or_number = [1, 2],
        then it will return the result of 1 <= value <= 2.
        Or range_or_number and percent could be range_or_number = 10,
        percent = 0.2
        then it will return the result of 10 * (1 - 0.2) <= value <= 10 * (1 + 0.2)
    '''
    the_range = range_or_number
    if percent > 0:
        the_range = (range_or_number*(1-percent), range_or_number*(1+percent))
    return the_range[0] <= value <= the_range[-1]

def cal_difference(the_list1, the_list2):
    return tuple((index, cur1, cur2) for index, (cur1, cur2) 
        in enumerate(zip(the_list1, the_list2)) if cur1!=cur2)
        


if __name__ == '__main__':
    from minitest import *

    with test(is_in_range):
        is_in_range(1.5, [1,2]).must_equal(True)
        is_in_range(3, [1,2]).must_equal(False)
        is_in_range(8, 10, 0.2).must_equal(True)
        is_in_range(6, 10, 0.2).must_equal(False)

    with test(cal_difference):
        the_list1 = (1,2,3,4,5,6)
        the_list2 = (1,2,5,4,7,6)
        cal_difference(the_list1, the_list2).must_equal(
            ((2, 3, 5), (4, 5, 7)))