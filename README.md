# Target

It will get the sudoku puzzle from the pic.

# Handling
## How to add supplement number:
1. change the file_path in generate_supplement_number_ragion of training_result_generator.py. like:
        file_path = '../../resource/svm_wrong_digits/pic16_no19_real5_cal1.dataset'
2. change the main method to run generate_supplement_number_ragion in training_result_generator.py
3. change the main method to run generate_supplement_result.
    notice, this step will over cover the old result files.
4. run the test in training_result_generator.py 
5. run the test in vertify.py

http://picturesudoku.herokuapp.com/

set
PYTHONPATH=/Users/colin/work/picture_sudoku

python          2.7.5

The python packages used:
numpy           1.6.2
opencv          2.4.9
bottle          0.12.7
minitest        1.3.2

The css packages used:
bootstrap       3.1.1
bootstrap-theme 3.1.1

The js packages used:
bootstrap   3.1.1
jquery      1.11.1