import requests

if __name__ == '__main__':
    from minitest import *

    with test("sudoku/webapp"):
        result = requests.get("http://localhost:9996/sudoku/webapp")
        result.content.must_equal('true')

    with test("/sudoku/image/result"):
        filename = '../resource/example_pics/sample01.dataset.jpg'    
        result = requests.post("http://localhost:9996/sudoku/image/result",
            files={'files':open(filename,'rb')})
        result.content.must_equal('{"pic_file_name": "sample01.dataset.jpg", "answered": {"6_4": 7, "6_5": 8, "6_7": 6, "6_0": 9, "8_0": 2, "6_2": 5, "6_3": 4, "1_8": 4, "8_1": 8, "6_8": 1, "1_7": 8, "7_5": 5, "1_5": 1, "1_4": 2, "8_2": 7, "0_8": 3, "0_7": 2, "0_6": 9, "7_4": 9, "7_7": 3, "0_2": 1, "7_0": 1, "7_3": 2, "6_1": 3, "2_8": 5, "3_0": 6, "3_3": 7, "3_2": 3, "3_5": 9, "3_6": 5, "2_0": 4, "3_8": 2, "2_3": 9, "2_4": 6, "2_5": 3, "2_6": 1, "2_7": 7, "5_3": 1, "5_2": 2, "5_0": 8, "5_6": 7, "5_5": 4, "4_2": 4, "7_1": 4, "5_8": 6, "4_6": 3, "4_4": 5, "8_6": 4, "2_1": 2, "1_3": 5, "1_1": 7}, "fixed": {"6_6": 2, "1_6": 6, "8_4": 1, "8_5": 6, "1_2": 9, "1_0": 3, "0_4": 4, "0_5": 7, "0_3": 8, "0_0": 5, "7_2": 6, "3_1": 1, "7_6": 8, "3_4": 8, "3_7": 4, "0_1": 6, "2_2": 8, "4_8": 8, "5_1": 5, "5_7": 9, "8_8": 9, "5_4": 3, "4_3": 6, "4_0": 7, "4_1": 9, "4_7": 1, "4_5": 2, "8_7": 5, "7_8": 7, "8_3": 3}, "status": "SUCCESS"}')

    with test("/sudoku/image/result"):
        filename = '../resource/example_pics/sample07.dataset.jpg'    
        result = requests.post("http://localhost:9996/sudoku/image/result",
            files={'files':open(filename,'rb')})
        result.content.must_equal('{"pic_file_name": "sample07.dataset.jpg", "answered": false, "fixed": {"8_0": 9, "1_0": 5, "1_7": 6, "8_7": 4, "8_4": 4, "1_4": 2, "7_8": 6, "1_1": 7, "8_1": 1, "0_7": 1, "7_7": 4, "7_1": 5, "0_4": 9, "0_0": 8, "0_1": 2, "3_3": 4, "3_5": 5, "3_4": 8, "3_8": 1, "7_0": 3, "5_3": 9, "5_0": 1, "8_8": 8, "5_5": 3, "5_4": 7, "1_8": 4, "0_8": 3, "7_4": 1, "5_8": 2}, "status": "FAILURE", "error": "Sorry, cannot answer it, since this puzzle may not follow the rules of Sudoku!"}')

    with test("/sudoku/image/result"):
        filename = '../resource/example_pics/false01.dataset.jpg'    
        result = requests.post("http://localhost:9996/sudoku/image/result",
            files={'files':open(filename,'rb')})
        result.content.must_equal('{"status": "FAILURE", "error": "\'NoneType\' object has no attribute \'__getitem__\'"}')
