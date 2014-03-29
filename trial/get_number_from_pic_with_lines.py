'''
    this method came from stackoverflow,
    and it cannot work very well for many pictures.
'''

import cv2
import numpy

from picture_sudoku.helpers import cv2_helper

show_pic = cv2_helper.show_pic


def image_close(gray_pic):
    ''' for closing which adjusts the brightness in the image, 
    by dividing each pixel with the result of a closing operation:'''
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    close = cv2.morphologyEx(gray_pic,cv2.MORPH_CLOSE,kernel1)
    div = numpy.float32(gray_pic)/(close)
    # show_pic(div)
    return numpy.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))

def find_max_contour(gray_pic):
    thresh = cv2.adaptiveThreshold(gray_pic,255,0,1,19,2)
    contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt
    return best_cnt

def get_masked_pic(gray_pic, contour):    
    mask = numpy.zeros((gray_pic.shape),numpy.uint8)


    cv2.drawContours(mask,[contour],contourIdx=0,color=255,thickness=-1)
    # show_pic(mask)
    cv2.drawContours(mask,[contour],contourIdx=0,color=0,thickness=2)
    # show_pic(mask)

    masked_pic = cv2.bitwise_and(gray_pic,mask)
    return masked_pic


def find_vertical_lines(gray_pic):
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

    dx = cv2.Sobel(gray_pic,cv2.CV_16S,1,0)
    dx.pp()
    # dx = cv2.Sobel(gray_pic,cv2.CV_32F,1,0)
    # convert from dtype=int16 to dtype=uint8
    dx = cv2.convertScaleAbs(dx)
    dx.pp()
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    # cv2_helper.show_pic(dx)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # close = cv2.adaptiveThreshold(dx,255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, blockSize=3, C=2)

    # cv2_helper.show_pic(close)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 8:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()
    # show_pic(closex)
    return closex

def find_horizontal_lines(gray_pic):
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    dy = cv2.Sobel(gray_pic,cv2.CV_16S,0,2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w/h > 8:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)

    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
    closey = close.copy()
    return closey

def find_points(gray_pic):
    contours, hier = cv2.findContours(gray_pic,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        mom = cv2.moments(cnt)
        if mom['m00'] == 0:
            continue
        # [mom['m10'],mom['m01'],mom['m00'] ].pp()
        (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
        # cv2.circle(img,(x,y),4,(0,255,0),-1)
        centroids.append((x,y))
    return centroids

def show_points_in_pic(pic_array, points):
    for point in points:
        cv2.circle(pic_array,point,4,(0,255,0),-1)
    show_pic(pic_array)


def main(image_path):
    image_path = '../resource/example_pics/sample14.dataset.jpg'
    img = cv2.imread(image_path)
    img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2_helper.resize_with_fixed_height(gray)
    img = cv2_helper.resize_with_fixed_height(img)
    # gray = cv2.imread(image_path, 0)
    # show_pic(gray)

    res = image_close(gray)
    show_pic(res)
    # res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
    # show_pic(res2)

    best_cnt = find_max_contour(res)

    len(best_cnt).pp()

    res = get_masked_pic(res, best_cnt)
    # show_pic(res)

    ''' 
        Finding Vertical lines
    '''
    closex = find_vertical_lines(res)
    show_pic(closex)

    closey = find_horizontal_lines(res)
    show_pic(closey)

    ''' Finding Grid Point areas'''
    res = cv2.bitwise_and(closex,closey)
    # show_pic(res)

    centroids = find_points(res)
    show_points_in_pic(img,centroids)




if __name__ == '__main__':
    from minitest import *
    # for i in range(1,15):
    #     pic_file_path = '../resource/example_pics/sample'+str(i).zfill(2)+'.dataset.jpg'
    #     main(pic_file_path)

    main('')
