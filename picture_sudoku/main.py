import numpy

from cv2_helpers.image import Image
from cv2_helpers.ragion import Ragion
from cv2_helpers.ragion import Ragions


from picture_analyzer import main_analyzer
from digit_recognition.multiple_svm import MultipleSvm
from digit_recognition.rbf_smo import Smo

IMG_SIZE = 32
FULL_SIZE = 1024


def transfer_to_digit_matrix(the_ragion):
    '''
        (1, FULL_SIZE) matrix
    '''
    heighted_ragion = Image.resize_keeping_ratio_by_height(the_ragion, IMG_SIZE)
    standard_ragion = Ragion.fill(heighted_ragion,(IMG_SIZE,IMG_SIZE))
    return numpy.matrix(standard_ragion.reshape(1, FULL_SIZE))


if __name__ == '__main__':
    from minitest import *
    from picture_sudoku.cv2_helpers.display import Display
    from picture_sudoku.helpers import numpy_helper
    from picture_sudoku.helpers import list_helper



    with test(vertify_all_pics):
        vertify_all_pics()
        # pic_01, 1, 4->1
        # pic_02, 1, 2->7
        # pic_03, 2, 4->1, 3->1
        # pic_04, 5, 4->1, 3->1, 6->5, 2->7, 5->1
        # pic_05, 1, 8->0
        # pic_06, 1, 5->6
        # pic_08, n
        # pic_09, 1, 7->1
        # pic_13, 3, 3->1, 2->7, 8->6
        # pic_14, 4, 8->7, 2->7, 6->7, 6->7, rotate
        # for i in range(1,15):
        #     pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
        #     pic_file_path.ppl()
        #     main(pic_file_path)
