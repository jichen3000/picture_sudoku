# it copys from Machine Learning in Action chapter 6

# sequential minimal optimization (SMO)
# John C. Platt, "Using Analytic QP and Sparseness 
# to Speed Training of Support Vector Machines"

# radial bias function,
# mapping from one feature space to another feature space.
# inner products.
# One great thing about the SVM optimization is that all 
# operations can be written in terms of inner products. 
# Inner products are two vectors multiplied together to 
# yield a scalar or single number.

# kernel trick or kernel substation.
# A popular kernel is the radial bias function, which we'll introduce next.

from functional_style import *

from functools import partial
from operator import itemgetter, gt, lt
from numpy import *
import random

import matplotlib.pyplot as plt

def get_dataset_from_file(filename):
    with open(filename) as datafile:
        words = [line.strip().split('\t') for line in datafile]
    dataset = [ [float(cell) for cell in row[:-1]] for row in words]
    labels = map(comb(itemgetter(-1), float), words)
    return dataset, labels

# select a value from 0 to m, but not equal the value of i
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# reset a value according to a range from low_value to hight_value
def clipAlpha(aj,hight_value,low_value):
    if aj > hight_value:
        aj = hight_value
    if aj < low_value:
        aj = low_value
    return aj

def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K /(-1*kTup[1]**2))
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*\
          oS.K[:,k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] 
    # oS.eCache.pp()
    # validEcacheList.pp()
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
       ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        # eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - \
        #         oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
             print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
        (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
             oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
             (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
             oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*\
             (oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, max_iteration, kTup=('lin', 0)): 
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup) 
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < max_iteration) and ((alphaPairsChanged > 0) \
            or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print "fullSet, iter: %d i:%d, pairs changed %d" %\
                    (iter,i,alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % \
                        (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas


def get_support_vectors(dataset, labels, alphas):
    # [item.pp() for item in zip(alphas, dataset, labels)]
    # return []
    return filter(lambda item: item[0]>0, 
        zip(alphas, dataset, labels))

def calcWs(dataArr, classLabels, alphas):
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMat)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],dataMat[i,:].T)
    return w

def testRbf(k1=1.3):
    dataArr,labelArr = get_dataset_from_file('test_set_RBF.dataset')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) 
    datMat=mat(dataArr); 
    labelMat = mat(labelArr).transpose() 
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m) 
    dataArr,labelArr = get_dataset_from_file('test_set_RBF2.dataset')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)

def draw_points(filename):
    def get_group_list(point_one_list, label_value):
        return [x for x,label in zip(point_one_list, labels) if label ==label_value]
    dataset,labels = get_dataset_from_file(filename)
    x_list, y_list = zip(*dataset)
    x_1_list = get_group_list(x_list, 1)
    x_0_list = get_group_list(x_list, -1)
    y_1_list = get_group_list(y_list, 1)
    y_0_list = get_group_list(y_list, -1)
    plt.plot(x_1_list, y_1_list, 'ro', x_0_list, y_0_list, 'bs')

def draw_all_points():
    draw_points("test_set_RBF.dataset")
    draw_points("test_set_RBF2.dataset")
    plt.show()


if __name__ == '__main__':
    from minitest import *

    with test_case("testRbf"):
        tself = get_test_self()

        with test("testRbf"):
            testRbf()
            pass

        with test("draw_all_points"):
            # draw_all_points()
            pass


