from bottle import route, run, request, static_file
import os
import json

@route("/css/<filename:path>")
def sudoku_css(filename):
    return static_file(filename, root='../../views/css')

@route("/js/<filename:path>")
def sudoku_js(filename):
    return static_file(filename, root='../../views/js')

@route("/")
def main_html():
    return static_file('upload_basic.html', root='./')

@route('/upload', method='POST')
def do_upload():
    upload = request.files['files']
    print dir(upload)
    # upload = request.files.get('upload')
    # name, ext = os.path.splitext(upload.filename)
    # if ext not in ('.png','.jpg','.jpeg'):
    #     return "File extension not allowed."

    save_path = "./"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    file_path = "{path}/{file}".format(path=save_path, file=upload.filename)
    file_path.ppl()
    if os.path.isfile(file_path):
        os.remove(file_path)
    upload.save(file_path)
    print 'file saved'
    # return "File successfully saved to '{0}'.".format(file_path)
    import cv2
    the_image = cv2.imread(file_path)
    the_image.shape.ppl()
    # cv2.imshow('123', the_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print 'after save!'
    # return file_path
    return json.dumps({'path':file_path})


if __name__ == '__main__':
    from minitest import *
    run(host='localhost', port=9999,reloader=True)
