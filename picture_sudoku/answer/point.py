ROW="row"
COL="col"
REGION="region"
NAMES=(ROW,COL,REGION)

def compute_region_index(row_index,col_index):
    return (row_index / 3) * 3 + (col_index / 3)

def is_same(point1,point2):
    return (point1[0]==point2[0] and point1[1]==point2[1])

def remove_points_from_first(full_list, sub_list):
#   result = full_list[:]
    for item in sub_list:
        full_list.remove(remove_value(item))
    return full_list

def remove_point_from_first(full_list, sub):
#   result = full_list[:]
    full_list.remove(remove_value(sub))
    return full_list

def intersect_list(pre_list, next_list):
    return [i for i in pre_list if i in next_list]


def add_value(cur_point,value):
    return cur_point+(value,)
def remove_value(cur_point):
    return cur_point[0:3]
def get_value(cur_point):
    return cur_point[3]
def get_row_index(cur_point):
    return cur_point[0]
def get_col_index(cur_point):
    return cur_point[1]

def _gen_point_with_row_col(row_index,col_index):
    return (row_index,col_index,
        compute_region_index(row_index,col_index))
def _gen_point_with_row_col_value(row_index,col_index,value):
    return (row_index,col_index,
        compute_region_index(row_index,col_index),
        value)

def gen_point(*args):
    '''
        return the point which has (x, y, ragion_index, value)
    '''
    if(len(args)==3):
        return _gen_point_with_row_col_value(
            args[0],args[1],args[2])
    elif(len(args)==2):
        return _gen_point_with_row_col_value(
            args[0][0],args[0][1],args[1])
    elif(len(args)==1):
        return _gen_point_with_row_col(args[0][0],args[0][1])

def gen_points(cur_point,values):
    return [add_value(cur_point,value) for value in values]

def create_point(total_index, value):
    y = total_index / 9
    x = total_index % 9
    return gen_point(x,y,value)

def is_points_duplicated(points):
    showed_numbers = {}
    for cur_point in points:
        cur_value = get_value(cur_point)
        for name, index in zip(NAMES,cur_point):
            if (name, index, cur_value) in showed_numbers.keys():
                return (showed_numbers[(name, index, cur_value)],cur_point)
            else:
                showed_numbers[(name, index, cur_value)] = cur_point
    return False

KEY_SPLIT = '_'
def transfer_point_key_value_to_point(key, value):
    return gen_point(map(int, key.split(KEY_SPLIT)),value)
def transfer_points_hash_to_points_list(points_hash):
    return [transfer_point_key_value_to_point(
        k,v) for k,v in points_hash.iteritems()]
def get_point_key(cur_point):
    return str(get_row_index(cur_point)
        )+KEY_SPLIT+str(get_col_index(cur_point))
def transfer_points_list_to_points_hash(points):
    return {get_point_key(cur_point): get_value(
        cur_point) for cur_point in points}
 
if __name__ == "__main__":
    from minitest import *

    with test(compute_region_index):
        compute_region_index(1,2).must_equal(0)

    with test(gen_point): 
        gen_point(1,2,4).must_equal((1, 2, 0, 4))
        gen_point((1,2)).must_equal((1, 2, 0))

    with test(intersect_list):
        intersect_list(range(2,5),range(3,9)).must_equal([3,4])

    with test(is_points_duplicated):
        is_points_duplicated(
            [(0, 4, 1, 9), (0, 5, 1, 8)]).must_false()
        is_points_duplicated(
            [(0, 4, 1, 8), (0, 5, 1, 8)]).must_equal(((0, 4, 1, 8), (0, 5, 1, 8)))

    with test(transfer_points_hash_to_points_list):
        transfer_points_hash_to_points_list({"0_1": 1, "7_8": 1}).must_equal(
            [(7, 8, 8, 1), (0, 1, 0, 1)])

    with test(transfer_points_list_to_points_hash):
        transfer_points_list_to_points_hash([(0, 4, 1, 9), (0, 5, 1, 8)]).must_equal(
            {'0_4': 9, '0_5': 8})

    with test(create_point):
        create_point(27,9).must_equal((0,3,1,9))
