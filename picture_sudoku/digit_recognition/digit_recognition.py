import os
from numpy import *

from rbf_smo import *

IMG_SIZE = 32

def img_to_vector(filename):
    with open(filename) as data_file:
        result = [int(line[index]) for line in data_file 
            for index in range(IMG_SIZE)]
    return result

def get_label_from_filename(filename):
    # filename.p()
    return filename.split('_')[0]

def get_label_and_data(pathname,filename):
    return get_label_from_filename(filename), img_to_vector(os.path.join(
            pathname, filename))

def get_handwriting_dataset(pathname):
    label_and_data_list = [get_label_and_data(pathname, filename) 
        for filename in os.listdir(pathname)]
    labels, dataset = zip(*label_and_data_list)
    return array(dataset), to_binary_labels(labels)

def to_binary_labels(labels):
    return [-1 if label=='9' else 1 for label in labels]

def testDigits(pathname, kTup=('rbf', 10)):
    dataArr,labelArr = get_handwriting_dataset(pathname)
    # dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup) 
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose() 
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m) 
    dataArr,labelArr = get_handwriting_dataset(pathname)
    # dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)
# def loadImages(dirName):
#     from os import listdir
#     hwLabels = []
#     trainingFileList = listdir(dirName)
#     m = len(trainingFileList)
#     trainingMat = zeros((m,1024))
#     for i in range(m):
#         fileNameStr = trainingFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         # fileStr.p()
#         classNumStr = int(fileStr.split('_')[0])
#         if classNumStr == 9: hwLabels.append(-1)
#         else: hwLabels.append(1)
#         trainingMat[i,:] = img_to_vector('%s/%s' % (dirName, fileNameStr))
#     return trainingMat, hwLabels

if __name__ == '__main__':
    from minitest import *
    with test_case("handwriting"):
        training_path = '../resource/training_digits'
        with test("get_handwriting_dataset"):
            # dataset1, labels1 = loadImages('../k_nearest_neighbours/test_digits')
            # dataset, labels = get_handwriting_dataset(training_path)
            # dataset.must_equal(dataset1, key=allclose)
            # labels.must_equal(labels1)
            pass

        with test("testDigits"):
            # it will take huge time, so please never run it again.
            # testDigits(training_path)
            pass

