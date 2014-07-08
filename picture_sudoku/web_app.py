import os
import sys

print "os.environ:",os.environ
sys.stdout.flush()


# just for heroku run
sys.path.append('/app/')

from bottle import route, run, request, static_file
from picture_sudoku import main_sudoku

@route("/manual_input")
def sudoku_main_html():
    return static_file('manual_input.html', root='views')

@route("/")
def sudoku_main_html():
    return static_file('picture_upload.html', root='views')

@route("/css/<filename:path>")
def sudoku_css(filename):
    return static_file(filename, root='views/css')

@route("/js/<filename:path>")
def sudoku_js(filename):
    return static_file(filename, root='views/js')

@route("/sudoku/webapp", method='GET')
def test_exists():
    return "true"


import json
# notice: when you use json, you must use post instead of get.
@route("/sudoku/input/result", method='POST')
def sudoku_result():
    points_hash = request.json
    # points_hash.pl()
    answer_result = main_sudoku.answer_quiz_with_point_hash(points_hash)
    if answer_result:
        return json.dumps(answer_result)
    else:
        return "false"

@route("/sudoku/image/result", method='POST')
def image_result():
    pic_file_path = save_upload_file(request.files['files'])
    answer_result = main_sudoku.answer_quiz_with_pic(pic_file_path)
    return json.dumps(answer_result)

def save_upload_file(upload_obj):
    # save_path = "resource/tmp_images/"
    save_path = "/tmp/tmp_images"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = "{path}/{file}".format(path=save_path, file=upload_obj.filename)
    # file_path.ppl()
    print "file_path:",os.path.abspath(file_path)
    sys.stdout.flush()
    if os.path.isfile(file_path):
        os.remove(file_path)
    upload_obj.save(file_path)
    return file_path  

# # for upload test
# @route('/upload/puzzle_image', method='POST')
# def do_upload():
#     print 'uploading...'
#     import time
#     time.sleep(2)
#     upload = request.files['files']
#     # print dir(upload)
#     # # upload = request.files.get('upload')
#     # # name, ext = os.path.splitext(upload.filename)
#     # # if ext not in ('.png','.jpg','.jpeg'):
#     # #     return "File extension not allowed."

#     # save_path = "resource/tmp_images/"
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)

#     # file_path = "{path}/{file}".format(path=save_path, file=upload.filename)
#     # file_path.ppl()
#     # if os.path.isfile(file_path):
#     #     os.remove(file_path)
#     # upload.save(file_path)
#     # return "File successfully saved to '{0}'.".format(file_path)
#     import cv2
#     # the_image = cv2.imread(file_path)
#     the_image = cv2.imread(upload)
#     the_image.shape.ppl()
#     # cv2.imshow('123', the_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     print 'after save!'
#     # return file_path
#     return json.dumps({'path':file_path})

# if __name__ == '__main__':
#     from minitest import *
    # print "http://localhost:9996"

if len(sys.argv) > 1:
    app_port = sys.argv[1]    
    run(host='0.0.0.0', port=app_port)
else:
    run(host='localhost', port=5000, reloader=True)
    # run(host='192.168.11.4', port=5000, reloader=True)
