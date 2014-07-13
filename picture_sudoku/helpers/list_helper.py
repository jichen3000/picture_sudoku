import operator

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
    ''' find the cal_different elements in two list which have the same length'''
    return tuple((index, cur1, cur2) for index, (cur1, cur2) 
        in enumerate(zip(the_list1, the_list2)) if cur1!=cur2)
        
def adjust_in_range(the_value, start_value, end_value):
    ''' change the value, let fall between start_value and end_value'''
    adjusted_value = max(the_value, start_value)
    adjusted_value = min(adjusted_value, end_value)
    return adjusted_value

def catalogue_list_list(the_list_list, the_index, threshold):
    get_value_func = operator.itemgetter(the_index)
    sorted_list_list = sorted(the_list_list, key=get_value_func)

    standard_value = get_value_func(sorted_list_list[0])
    catalogued_list = [ [sorted_list_list[0]] ]
    cur_index = 0
    for cur_list  in sorted_list_list[1::]:
        cur_value = get_value_func(cur_list)
        if standard_value + threshold > cur_value:
            catalogued_list[cur_index].append(cur_list)
        else:
            cur_index += 1
            catalogued_list  += [[cur_list]]
        standard_value = cur_value
    return catalogued_list


if __name__ == '__main__':
    from minitest import *
    import numpy

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

    with test(adjust_in_range):
        adjust_in_range(the_value=4, start_value=0, end_value=10).must_equal(4)
        adjust_in_range(the_value=0, start_value=0, end_value=10).must_equal(0)
        adjust_in_range(the_value=10, start_value=0, end_value=10).must_equal(10)
        adjust_in_range(the_value=-1, start_value=0, end_value=10).must_equal(0)
        adjust_in_range(the_value=11, start_value=0, end_value=10).must_equal(10)

    with test(catalogue_list_list):
        lines = [numpy.array([ 214.        ,    1.57079637]),
                 numpy.array([ 319.        ,    1.57079637]),
                 numpy.array([ 10.        ,   1.57079637]),
                 numpy.array([ 6.        ,  1.57079637]),
                 numpy.array([ 13.        ,   1.57079637]),
                 numpy.array([ 221.        ,    1.57079637]),
                 numpy.array([ 326.        ,    1.57079637]),
                 numpy.array([ 217.        ,    1.58824956]),
                 numpy.array([ 113.        ,    1.58824956]),
                 numpy.array([ 44.        ,   1.57079637]),
                 numpy.array([ 5.        ,  1.58824956]),
                 numpy.array([ 16.        ,   1.55334306]),
                 numpy.array([ 322.        ,    1.58824956]),
                 numpy.array([ 219.        ,    1.55334306]),
                 numpy.array([ 117.        ,    1.55334306]),
                 numpy.array([ 318.        ,    1.58824956]),
                 numpy.array([ 110.        ,    1.58824956]),
                 numpy.array([ 14.        ,   1.53588974]),
                 numpy.array([ 114.        ,    1.57079637]),
                 numpy.array([ 9.        ,  1.55334306]),
                 numpy.array([ 324.        ,    1.55334306]),
                 numpy.array([ 224.        ,    1.53588974]),
                 numpy.array([ 329.        ,    1.55334306]),
                 numpy.array([ 11.        ,   1.55334306]),
                 numpy.array([ 111.        ,    1.57079637])]
        accuracy_pixs = 330 / 9 *0.5 # 9
        # accuracy_pixs.ppl()
        catalogued_lines = catalogue_list_list(lines, 0, accuracy_pixs)
        catalogued_lines.size().must_equal(5)
        catalogued_lines.ppl()
