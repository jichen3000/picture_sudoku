import point
import copy

def gen_99():
    return [(i,j) for i in range(9) for j in range(9)]

def remove_number(unshow_numbers, known_point):
    row_index, col_index, region_index, value = known_point
    unshow_numbers[(point.ROW, row_index)].remove(value)
    unshow_numbers[(point.COL, col_index)].remove(value)
    unshow_numbers[(point.REGION, region_index)].remove(value)
    return unshow_numbers

def remove_numbers(unshow_numbers, known_points):
    return reduce(remove_number, known_points, unshow_numbers)

def get_intersect_values(cur_point,unshow_numbers):
    values_list = map(lambda name, index: unshow_numbers[(name, index)],
            point.NAMES, cur_point)
    return reduce(point.intersect_list,values_list)


def gen_known_points_from_sample(sample):
    # def append_point(result,cur_point):
    #     row_index, col_index = cur_point
    #     value = sample[row_index][col_index]
    #     if value > 0:
    #         result.append(point.gen_point(row_index,col_index,value))
    #     return result
    # return reduce(append_point, gen_99(), [])
    return [point.gen_point(row_index,col_index,sample[row_index][col_index]) 
        for row_index, col_index in gen_99() 
        if sample[row_index][col_index] > 0]

def gen_unknow_points(known_points):
    unknow_points = map(point.gen_point,gen_99())
    return point.remove_points_from_first(unknow_points, known_points)

def gen_unshow_numbers(known_points):
    def append_number(result, cur_point):
        map(lambda name, index: result.update({(name, index):range(1,10)}),
            point.NAMES, point.gen_point(cur_point))
        return result
    initial_unshow_numbers = reduce(append_number, gen_99(), {})
    return remove_numbers(initial_unshow_numbers,known_points)

# notice the side effect of this method
def exclude_compute_once(unknow_points, unshow_numbers):
    def compute_point(result, cur_point):
        values = get_intersect_values(cur_point,unshow_numbers)
        if(len(values)==1):
            cur_valued_point = point.add_value(cur_point,values.pop())
            result.append(cur_valued_point)
            remove_number(unshow_numbers, cur_valued_point)
        return result
    return reduce(compute_point, unknow_points, [])

def exclude_compute_all(unknow_points, unshow_numbers):
    computed_points = exclude_compute_once(unknow_points, unshow_numbers)
    #remain_unshow_numbers = remove_numbers(unshow_numbers,computed_points)
    if len(computed_points)==0:
        return []
    else:   
        return computed_points+exclude_compute_all(
            point.remove_points_from_first(unknow_points, computed_points), 
            unshow_numbers)

def choose_guess_point_values(unknow_points,unshow_numbers):
    choosed_point = min(unknow_points, 
        key=lambda cur_point: 
        len(get_intersect_values(cur_point,unshow_numbers)))
    return point.gen_points(choosed_point,
        get_intersect_values(choosed_point,unshow_numbers))

def ormap(fun, iter):
    for i in iter:
        fun_result = fun(i)
        if fun_result:
            return fun_result
    return False


def main_answer_quiz(known_points):
    if point.is_points_duplicated(known_points):
        return False
    computed_points = answer(gen_unknow_points(known_points),
        gen_unshow_numbers(known_points))
    if computed_points and (not point.is_points_duplicated(
        known_points+computed_points)):
        return point.transfer_points_list_to_points_hash(computed_points)
    return False

def answer_quiz_with_point_hash(points_hash):
    known_points = point.transfer_points_hash_to_points_list(points_hash)
    return main_answer_quiz(known_points)

def answer_quiz_with_indexs_and_digits(number_indexs, digits):
    def create_points(number_indexs, digits):
        return map(point.create_point, number_indexs, digits)
    known_points = create_points(number_indexs, digits)
    result = {'fixed':point.transfer_points_list_to_points_hash(known_points),
              'answered':main_answer_quiz(known_points)}
    return result

def answer(unknow_points, unshow_numbers):
    computed_points = exclude_compute_all(unknow_points, unshow_numbers)
    remain_unknow_points = unknow_points
    remain_unshow_numbers = unshow_numbers
    def II(guess_point):
        answer_result = answer(
            point.remove_point_from_first(
                remain_unknow_points[:],guess_point),
            # notice: must use deepcopy instead of dict.copy(), since the later is shalow copy.
            remove_number(copy.deepcopy(remain_unshow_numbers), guess_point))
        if answer_result :
            return computed_points+[guess_point,]+answer_result
        else:
            return False
    if len(remain_unknow_points)==0:
        return computed_points
    else:
        guess_points = choose_guess_point_values(
            reversed(remain_unknow_points),remain_unshow_numbers)
        return ormap(II, guess_points)

if __name__ == '__main__':
    from minitest import *
    excludable_sample = (
                (0,0,0,3,0,5,0,0,0),
                (0,0,2,0,9,0,3,0,0),
                (3,0,0,0,7,0,0,0,1),
                (5,0,0,0,1,0,0,0,9),
                (0,0,6,0,8,0,5,0,0),
                (0,0,0,5,0,2,0,0,0),
                (7,0,8,1,4,9,6,0,3),
                (0,9,0,0,0,0,0,2,0),
                (4,0,3,6,2,7,9,0,8))
    media_sample = (
                (0,0,0,0,9,8,0,2,0),
                (0,0,0,2,0,0,1,0,4),
                (0,0,0,0,0,6,5,0,0),
                (6,0,0,0,4,0,0,9,0),
                (0,0,0,8,0,3,6,0,0),
                (4,0,0,0,0,0,0,0,0),
                (7,0,9,3,2,0,0,0,5),
                (0,0,1,0,0,7,0,0,0),
                (0,2,0,0,0,0,7,0,0))
    difficult_sample = (
                (9,0,0,0,0,0,0,0,5),
                (0,4,0,3,0,0,0,2,0),
                (0,0,8,0,0,0,1,0,0),
                (0,7,0,6,0,3,0,0,0),
                (0,0,0,0,8,0,0,0,0),
                (0,0,0,7,0,9,0,6,0),
                (0,0,1,0,0,0,9,0,0),
                (0,3,0,0,0,6,0,4,0),
                (5,0,0,0,0,0,0,0,8))

    with test(gen_known_points_from_sample):
        known_points = gen_known_points_from_sample(media_sample)
        known_points.must_equal(
            [(0, 4, 1, 9), (0, 5, 1, 8), (0, 7, 2, 2), (1, 3, 1, 2), 
             (1, 6, 2, 1), (1, 8, 2, 4), (2, 5, 1, 6), (2, 6, 2, 5), 
             (3, 0, 3, 6), (3, 4, 4, 4), (3, 7, 5, 9), (4, 3, 4, 8), 
             (4, 5, 4, 3), (4, 6, 5, 6), (5, 0, 3, 4), (6, 0, 6, 7), 
             (6, 2, 6, 9), (6, 3, 7, 3), (6, 4, 7, 2), (6, 8, 8, 5), 
             (7, 2, 6, 1), (7, 5, 7, 7), (8, 1, 6, 2), (8, 6, 8, 7)])

        # difficult_known_points = gen_known_points_from_sample(difficult_sample)
        # point.transfer_points_list_to_points_hash(difficult_known_points).pp()


    with test(gen_unknow_points):
        unknow_points = gen_unknow_points(known_points)
        unknow_points.size().must_equal(57)
        unknow_points.must_equal(
            [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 1), (0, 6, 2), (0, 8, 2), 
             (1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 4, 1), (1, 5, 1), (1, 7, 2), 
             (2, 0, 0), (2, 1, 0), (2, 2, 0), (2, 3, 1), (2, 4, 1), (2, 7, 2), 
             (2, 8, 2), (3, 1, 3), (3, 2, 3), (3, 3, 4), (3, 5, 4), (3, 6, 5), 
             (3, 8, 5), (4, 0, 3), (4, 1, 3), (4, 2, 3), (4, 4, 4), (4, 7, 5), 
             (4, 8, 5), (5, 1, 3), (5, 2, 3), (5, 3, 4), (5, 4, 4), (5, 5, 4), 
             (5, 6, 5), (5, 7, 5), (5, 8, 5), (6, 1, 6), (6, 5, 7), (6, 6, 8), 
             (6, 7, 8), (7, 0, 6), (7, 1, 6), (7, 3, 7), (7, 4, 7), (7, 6, 8), 
             (7, 7, 8), (7, 8, 8), (8, 0, 6), (8, 2, 6), (8, 3, 7), (8, 4, 7), 
             (8, 5, 7), (8, 7, 8), (8, 8, 8)])



    with test(gen_unshow_numbers):
        unshow_numbers = gen_unshow_numbers(known_points)
        unshow_numbers.must_equal(
            {('col', 6): [2, 3, 4, 8, 9], 
             ('region', 8): [1, 2, 3, 4, 6, 8, 9], 
             ('region', 5): [1, 2, 3, 4, 5, 7, 8], 
             ('region', 6): [3, 4, 5, 6, 8], 
             ('region', 1): [1, 3, 4, 5, 7], 
             ('col', 8): [1, 2, 3, 6, 7, 8, 9], 
             ('row', 4): [1, 2, 4, 5, 7, 9], 
             ('row', 6): [1, 4, 6, 8], 
             ('col', 1): [1, 3, 4, 5, 6, 7, 8, 9], 
             ('col', 3): [1, 4, 5, 6, 7, 9], 
             ('row', 3): [1, 2, 3, 5, 7, 8], 
             ('col', 5): [1, 2, 4, 5, 9], 
             ('row', 1): [3, 5, 6, 7, 8, 9], 
             ('col', 7): [1, 3, 4, 5, 6, 7, 8], 
             ('region', 0): [1, 2, 3, 4, 5, 6, 7, 8, 9], 
             ('region', 4): [1, 2, 5, 6, 7, 9], 
             ('row', 0): [1, 3, 4, 5, 6, 7], 
             ('row', 5): [1, 2, 3, 5, 6, 7, 8, 9], 
             ('col', 0): [1, 2, 3, 5, 8, 9], 
             ('row', 8): [1, 3, 4, 5, 6, 8, 9], 
             ('region', 7): [1, 4, 5, 6, 8, 9], 
             ('col', 2): [2, 3, 4, 5, 6, 7, 8], 
             ('row', 2): [1, 2, 3, 4, 7, 8, 9], 
             ('row', 7): [2, 3, 4, 5, 6, 8, 9], 
             ('col', 4): [1, 3, 5, 6, 7, 8], 
             ('region', 2): [3, 6, 7, 8, 9], 
             ('region', 3): [1, 2, 3, 5, 7, 8, 9]})

    with test(answer_quiz_with_point_hash):
        point_hash = {  u'0_0': 5, u'0_1': 6, u'0_3': 8, u'0_4': 4, u'0_5': 7, 
                        u'1_0': 3, u'1_2': 9, u'1_6': 6, u'2_2': 8, u'3_1': 1, 
                        u'3_4': 8, u'3_7': 4, u'4_0': 7, u'4_1': 9, u'4_3': 6, 
                        u'4_5': 2, u'4_7': 1, u'4_8': 8, u'5_1': 5, u'5_4': 3, 
                        u'5_7': 9, u'6_6': 2, u'7_2': 6, u'7_6': 8, u'7_8': 7, 
                        u'8_3': 3, u'8_4': 1, u'8_5': 6, u'8_7': 5, u'8_8': 9 }
        answer_quiz_with_point_hash(point_hash).must_equal(
            {'0_2': 1, '0_6': 9, '0_7': 2, '0_8': 3, '1_1': 7,
             '1_3': 5, '1_4': 2, '1_5': 1, '1_7': 8, '1_8': 4,
             '2_0': 4, '2_1': 2, '2_3': 9, '2_4': 6, '2_5': 3,
             '2_6': 1, '2_7': 7, '2_8': 5, '3_0': 6, '3_2': 3,
             '3_3': 7, '3_5': 9, '3_6': 5, '3_8': 2, '4_2': 4,
             '4_4': 5, '4_6': 3, '5_0': 8, '5_2': 2, '5_3': 1,
             '5_5': 4, '5_6': 7, '5_8': 6, '6_0': 9, '6_1': 3, 
             '6_2': 5, '6_3': 4, '6_4': 7, '6_5': 8, '6_7': 6, 
             '6_8': 1, '7_0': 1, '7_1': 4, '7_3': 2, '7_4': 9, 
             '7_5': 5, '7_7': 3, '8_0': 2, '8_1': 8, '8_2': 7, 
             '8_6': 4 })

    with test(answer_quiz_with_indexs_and_digits):
        number_indexs = (0, 1, 4, 9, 12, 13, 14, 19, 20, 25, 27, 31, 35, 36, 39, 
            41, 44, 45, 49, 53, 55, 60, 61, 66, 67, 68, 71, 76, 79, 80)
        digits = (5, 3, 7, 6, 1, 9, 5, 9, 8, 6, 8, 6, 3, 4, 8, 
            3, 1, 7, 2, 6, 6, 2, 8, 4, 1, 9, 5, 8, 7, 9)
        answer_quiz_with_indexs_and_digits(number_indexs, digits).must_equal(
            { 'fixed': 
                   {'0_0': 5, '0_1': 6, '0_3': 8, '0_4': 4, '0_5': 7, 
                    '1_0': 3, '1_2': 9, '1_6': 6, '2_2': 8, '3_1': 1, 
                    '3_4': 8, '3_7': 4, '4_0': 7, '4_1': 9, '4_3': 6, 
                    '4_5': 2, '4_7': 1, '4_8': 8, '5_1': 5, '5_4': 3, 
                    '5_7': 9, '6_6': 2, '7_2': 6, '7_6': 8, '7_8': 7, 
                    '8_3': 3, '8_4': 1, '8_5': 6, '8_7': 5, '8_8': 9},
              'answered':
                   {'0_2': 1, '0_6': 9, '0_7': 2, '0_8': 3, '1_1': 7,
                    '1_3': 5, '1_4': 2, '1_5': 1, '1_7': 8, '1_8': 4,
                    '2_0': 4, '2_1': 2, '2_3': 9, '2_4': 6, '2_5': 3,
                    '2_6': 1, '2_7': 7, '2_8': 5, '3_0': 6, '3_2': 3,
                    '3_3': 7, '3_5': 9, '3_6': 5, '3_8': 2, '4_2': 4,
                    '4_4': 5, '4_6': 3, '5_0': 8, '5_2': 2, '5_3': 1,
                    '5_5': 4, '5_6': 7, '5_8': 6, '6_0': 9, '6_1': 3, 
                    '6_2': 5, '6_3': 4, '6_4': 7, '6_5': 8, '6_7': 6, 
                    '6_8': 1, '7_0': 1, '7_1': 4, '7_3': 2, '7_4': 9, 
                    '7_5': 5, '7_7': 3, '8_0': 2, '8_1': 8, '8_2': 7, 
                    '8_6': 4 }})


        # answer_quiz_with_indexs_and_digits(number_indexs, digits).must_equal(
        #     {'0_2': 1, '0_6': 9, '0_7': 2, '0_8': 3, '1_1': 7,
        #      '1_3': 5, '1_4': 2, '1_5': 1, '1_7': 8, '1_8': 4,
        #      '2_0': 4, '2_1': 2, '2_3': 9, '2_4': 6, '2_5': 3,
        #      '2_6': 1, '2_7': 7, '2_8': 5, '3_0': 6, '3_2': 3,
        #      '3_3': 7, '3_5': 9, '3_6': 5, '3_8': 2, '4_2': 4,
        #      '4_4': 5, '4_6': 3, '5_0': 8, '5_2': 2, '5_3': 1,
        #      '5_5': 4, '5_6': 7, '5_8': 6, '6_0': 9, '6_1': 3, 
        #      '6_2': 5, '6_3': 4, '6_4': 7, '6_5': 8, '6_7': 6, 
        #      '6_8': 1, '7_0': 1, '7_1': 4, '7_3': 2, '7_4': 9, 
        #      '7_5': 5, '7_7': 3, '8_0': 2, '8_1': 8, '8_2': 7, 
        #      '8_6': 4 })

    with test(choose_guess_point_values):
        choose_guess_point_values(unknow_points, unshow_numbers).must_equal([(0, 6, 2, 3)])

    with test(point.is_points_duplicated):
        computed_all_points = answer(
            unknow_points[:], copy.deepcopy(unshow_numbers))
        point.is_points_duplicated(known_points+computed_all_points).must_false()
        computed_all_points.size().must_equal(57)

    with test(get_intersect_values):
        get_intersect_values((8,4,7),unshow_numbers).must_equal([1, 5, 6, 8])
