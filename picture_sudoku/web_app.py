from bottle import route, run, request, static_file
from picture_sudoku import main_sudoku


@route("/manual_input")
def sudoku_main_html():
    return static_file('manual_input.html', root='../views')

@route("/")
def sudoku_main_html():
    return static_file('picture_upload.html', root='../views')

@route("/css/<filename:path>")
def sudoku_css(filename):
    return static_file(filename, root='../views/css')

@route("/js/<filename:path>")
def sudoku_js(filename):
    return static_file(filename, root='../views/js')

import json
# notice: when you use json, you must use post instead of get.
@route("/sudoku/sudokuresult", method='POST')
def sudoku_result():
    points_hash = request.json
    answer = main_sudoku.answer_quiz_with_point_hash(points_hash)
    if answer:
        return json.dumps(answer)
    else:
        return "false"


# print "http://localhost:9996"
run(host='localhost', port=9996,reloader=True)
