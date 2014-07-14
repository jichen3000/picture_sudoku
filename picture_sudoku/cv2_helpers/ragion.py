import cv2
import numpy
import operator

BLACK = 0
WHITE = 255

class Ragion(object):
    ''' Actually it is just a part of image. '''                

    @staticmethod
    def fill(the_ragion, target_shape, fill_value=0):
        '''
            Ragion.fill
            Fill the ragion to the size of target_shape.
        '''
        height, width = the_ragion.shape        
        target_height, target_width = target_shape

        if height > target_height and width > target_width:
            return the_ragion

        cur_ragion = the_ragion.copy()
        if target_height > height:
            top_height = (target_height - height) / 2
            bottom_height = target_height - top_height - height

            top_mat = numpy.zeros((top_height, width)) + fill_value
            bottom_mat = numpy.zeros((bottom_height, width)) + fill_value

            cur_ragion = numpy.concatenate((top_mat, cur_ragion, bottom_mat), axis=0)

            # cur_ragion.pp()
        if target_width > width:
            left_width = (target_width - width) / 2
            right_width = target_width - left_width - width

            right_mat = numpy.zeros((target_height, right_width)) + fill_value
            left_mat = numpy.zeros((target_height, left_width)) + fill_value

            cur_ragion = numpy.concatenate((left_mat, cur_ragion, right_mat), axis=1)
        return cur_ragion

    @staticmethod
    def review_classified_number_ragion_for_8(the_ragion, the_digit):
        '''
            change the digit which has been recognized as 8,
            but actually is the other ones, like 6. 
        '''
        if the_digit != 8 :
            return the_digit
        height, width = the_ragion.shape
        def review_for_6():
            # for 6, check the top right part
            # is there a line which is all 0
            end_y = height / 2
            start_y = height / 4
            start_x = width / 2
            end_x = width
            
            for y_index in range(end_y+1, start_y, -1):
                whole_half_line_is_0 = True
                for x_index in range(start_x, end_x):
                    if(the_ragion[y_index, x_index] > 0):
                        whole_half_line_is_0 = False
                        break
                if whole_half_line_is_0:
                    return 6
            return 8
        return review_for_6()

class Ragions(object):
    @staticmethod
    def join(ragions, count_in_row=9, init_value=BLACK):
        return Ragions.join_same_size(
            Ragions.fill_to_same_size(ragions),count_in_row, init_value)

    @staticmethod
    def fill_to_same_size(ragions):
        '''
            Ragions.fill_to_same_size
            Fill all ragions to the same size which is the largest in height and width.
        '''
        shapes = map(numpy.shape, ragions)
        largest_height = max(map(operator.itemgetter(0), shapes))
        largest_width = max(map(operator.itemgetter(1), shapes))
        fill_func = lambda ragion: Ragion.fill(ragion, (largest_height, largest_width))
        return map(fill_func, ragions)

    @staticmethod
    def join_same_size(ragions, count_in_row=9, init_value=BLACK):
        '''
            Ragions.join_same_size
            Join all ragions to a big image.
            The ragions must have same size.
        '''
        ragion_count = len(ragions)
        # ragion_count.pp()
        ragion_row_count = int(ragion_count / count_in_row) + 1
        steps = 4
        ragion_height, ragion_width = ragions[0].shape
        # ragions[0].shape.pp()
        width = count_in_row * ragion_width + (count_in_row + 1) * steps
        height = ragion_row_count * ragion_height + (ragion_row_count + 1) * steps

        pic_array = numpy.zeros((height, width), dtype=numpy.uint8) + init_value
        # pic_array.shape.pp()
        # show_pic(pic_array)

        for i in range(ragion_row_count):
            for j in range(count_in_row):
                ragion_index = j+i * count_in_row
                if ragion_index >= ragion_count:
                    break
                x_index = (i+1)*steps+i*ragion_height
                y_index = (j+1)*steps+j*ragion_width
                pic_array[x_index:x_index+ragion_height, y_index:y_index+ragion_width] = ragions[ragion_index]
        return pic_array



if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.image import Image
    from picture_sudoku.cv2_helpers.display import Display

    inject(numpy.allclose, 'must_close')

    with test("Ragion.fill"):
        the_ragion = numpy.ones((3,4))
        Ragion.fill(the_ragion, (6,6)).must_close(
            numpy.array(  [[ 0.,  0.,  0.,  0.,  0.,  0.],
                           [ 0.,  1.,  1.,  1.,  1.,  0.],
                           [ 0.,  1.,  1.,  1.,  1.,  0.],
                           [ 0.,  1.,  1.,  1.,  1.,  0.],
                           [ 0.,  0.,  0.,  0.,  0.,  0.],
                           [ 0.,  0.,  0.,  0.,  0.,  0.]]))

    with test(Ragion.review_classified_number_ragion_for_8):
        the_pic_path = '../../resource/test/pic17_no08_real6_cal8.dataset'
        the_ragion = Image.load_from_txt(the_pic_path)
        Ragion.review_classified_number_ragion_for_8(the_ragion, 8).must_equal(6)


        the_pic_path = '../../resource/test/pic16_no05_real8_cal6.dataset'
        the_ragion = Image.load_from_txt(the_pic_path)
        Ragion.review_classified_number_ragion_for_8(the_ragion, 8).must_equal(8)
        # Display.binary_image(the_ragion)
        # save file: ../resource/svm_wrong_digits/pic16_no10_real8_cal6.dataset
        # save file: ../resource/svm_wrong_digits/pic16_no20_real8_cal6.dataset
        pass

    with test("Ragions.fill_to_same_size"):
        ragions = (numpy.ones((3,2)), numpy.ones((2,1)), numpy.ones((2,4)))
        Ragions.fill_to_same_size(ragions).must_close(
            [numpy.array([[ 0.,  1.,  1.,  0.],
                   [ 0.,  1.,  1.,  0.],
                   [ 0.,  1.,  1.,  0.]]),
             numpy.array([[ 0.,  1.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.]]),
             numpy.array([[ 1.,  1.,  1.,  1.],
                   [ 1.,  1.,  1.,  1.],
                   [ 0.,  0.,  0.,  0.]])])
