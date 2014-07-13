import requests
import json

from picture_sudoku.helpers.common import Resource, OtherResource

if __name__ == '__main__':
    from minitest import *

    # SERVER_ADDRESS = "http://localhost:5000"
    SERVER_ADDRESS = "http://192.168.11.4:5000"
    IMAGE_RESULT_URL = SERVER_ADDRESS + "/sudoku/image/result"

    with test("sudoku/webapp"):
        result = requests.get(SERVER_ADDRESS + "/sudoku/webapp")
        result.content.must_equal('true')

    # with test("/sudoku/image/result"):
    #     pic_filepath = Resource.get_path('example_pics/sample01.dataset.jpg')
    #     # pic_filepath = '../resource/example_pics/sample01.dataset.jpg'    
    #     result = requests.post(IMAGE_RESULT_URL,
    #         files={'files':open(pic_filepath,'rb')})
    #     answer_result = json.loads(result.content)
    #     answer_result.must_equal(
    #         { 'fixed': 
    #                {'0_0': 5, '0_1': 6, '0_3': 8, '0_4': 4, '0_5': 7, 
    #                 '1_0': 3, '1_2': 9, '1_6': 6, '2_2': 8, '3_1': 1, 
    #                 '3_4': 8, '3_7': 4, '4_0': 7, '4_1': 9, '4_3': 6, 
    #                 '4_5': 2, '4_7': 1, '4_8': 8, '5_1': 5, '5_4': 3, 
    #                 '5_7': 9, '6_6': 2, '7_2': 6, '7_6': 8, '7_8': 7, 
    #                 '8_3': 3, '8_4': 1, '8_5': 6, '8_7': 5, '8_8': 9},
    #           'answered':
    #                {'0_2': 1, '0_6': 9, '0_7': 2, '0_8': 3, '1_1': 7,
    #                 '1_3': 5, '1_4': 2, '1_5': 1, '1_7': 8, '1_8': 4,
    #                 '2_0': 4, '2_1': 2, '2_3': 9, '2_4': 6, '2_5': 3,
    #                 '2_6': 1, '2_7': 7, '2_8': 5, '3_0': 6, '3_2': 3,
    #                 '3_3': 7, '3_5': 9, '3_6': 5, '3_8': 2, '4_2': 4,
    #                 '4_4': 5, '4_6': 3, '5_0': 8, '5_2': 2, '5_3': 1,
    #                 '5_5': 4, '5_6': 7, '5_8': 6, '6_0': 9, '6_1': 3, 
    #                 '6_2': 5, '6_3': 4, '6_4': 7, '6_5': 8, '6_7': 6, 
    #                 '6_8': 1, '7_0': 1, '7_1': 4, '7_3': 2, '7_4': 9, 
    #                 '7_5': 5, '7_7': 3, '8_0': 2, '8_1': 8, '8_2': 7, 
    #                 '8_6': 4 },
    #           'pic_file_name': 'sample01.dataset.jpg',
    #           'status': 'SUCCESS'})

    def test_the_sample(sample_index):
        pic_filepath = Resource.get_path('example_pics/sample'+
            str(sample_index).zfill(2)+'.dataset.jpg')
        # pic_filepath = '../resource/example_pics/sample01.dataset.jpg'    
        result = requests.post(IMAGE_RESULT_URL,
            files={'files':open(pic_filepath,'rb')})
        answer_result = json.loads(result.content)
        return answer_result

    with test("/sudoku/image/result"):
        for i in range(2,7) + range(8,15):
            answer_result = test_the_sample(i)
            answer_result['status'].must_equal('SUCCESS')
            answer_result['pic_file_name'].p()
            if answer_result['status'] != 'SUCCESS':
                answer_result.pp()
        # test_the_sample(8).pp()
        # test_the_sample(6).pp()
        
    # with test("/sudoku/image/result"):
    #     pic_filepath = '../resource/example_pics/sample07.dataset.jpg'    
    #     result = requests.post(IMAGE_RESULT_URL,
    #         files={'files':open(pic_filepath,'rb')})
    #     answer_result = json.loads(result.content)
    #     ERROR_CANNOT_ANSWER = "Sorry, cannot answer it, since this puzzle may not follow the rules of Sudoku!"
    #     answer_result.must_equal(
    #         {'pic_file_name': 'sample07.dataset.jpg', 
    #          'status': 'FAILURE', 
    #          'error': ERROR_CANNOT_ANSWER,
    #          'fixed': 
    #                 {'8_0': 9, '1_0': 5, '1_7': 6, '8_7': 4, '8_4': 4, 
    #                  '1_4': 2, '7_8': 6, '1_1': 7, '8_1': 1, '0_7': 1, 
    #                  '7_7': 4, '7_1': 5, '0_4': 9, '0_0': 8, '0_1': 2, 
    #                  '3_3': 4, '3_5': 5, '3_4': 8, '3_8': 1, '7_0': 3, 
    #                  '5_3': 9, '8_8': 8, '5_5': 3, '5_4': 7, 
    #                  '1_8': 4, '0_8': 3, '7_4': 1, '5_8': 2}
    #         })


    # with test("/sudoku/image/result"):
    #     pic_filepath = '../resource/example_pics/false01.dataset.jpg'    
    #     result = requests.post(IMAGE_RESULT_URL,
    #         files={'files':open(pic_filepath,'rb')})
    #     answer_result = json.loads(result.content)
    #     answer_result.must_equal(
    #         {'error': "Cannot find sudoku square!",
    #          'pic_file_name': 'false01.dataset.jpg',
    #          'status': 'FAILURE'})

    # with test("for issues"):
    #     pic_filepath = '../resource/for_issues/cannot_recognize.jpg'    
    #     result = requests.post(IMAGE_RESULT_URL,
    #         files={'files':open(pic_filepath,'rb')})
    #     answer_result = json.loads(result.content)
    #     answer_result.ppl()
