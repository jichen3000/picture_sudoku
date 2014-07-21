'''
from picture_sudoku.helpers.common import Resource, OtherResource
'''

import os

class Resource(object):
    resource_relative_path = '../../resource'
    @classmethod
    def get_path(cls, file_path=None, *others):
        cur_path = os.path.dirname(__file__)
        resource_path = os.path.join(cur_path, cls.resource_relative_path)
        if not file_path:
            result = os.path.abspath(resource_path)
        else:
            result =  os.path.abspath(
                os.path.join(resource_path, file_path))
        if others:
            result = os.path.join(result, *others)
        return result

class OtherResource(Resource):
    resource_relative_path = '../../other_resource'

if __name__ == '__main__':
    from minitest import *

    with test(Resource.get_path):
        resource_path = Resource.get_path()
        # resource_path.must_equal('/Users/colin/work/picture_sudoku/resource')
        os.path.exists(resource_path).must_true()

        sample_01_path = Resource.get_path('example_pics/sample01.dataset.jpg')
        os.path.exists(sample_01_path).must_true()

        sample_01_path = Resource.get_path('example_pics','sample01.dataset.jpg')
        os.path.exists(sample_01_path).must_true()


    with test(OtherResource.get_path):
        other_resource_path = OtherResource.get_path()
        # other_resource_path.must_equal('/Users/colin/work/picture_sudoku/other_resource')
        os.path.exists(other_resource_path).must_true()

        multiple_binary_path = OtherResource.get_path('font_training_result/multiple_binary.finaldata')
        os.path.exists(multiple_binary_path).must_true()
