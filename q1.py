import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

def pad(img, p):
    hei, wid, chan = img.shape
    imgPad = np.zeros((hei + 2 * p, wid + 2 * p, chan), img.dtype)
    imgPad[p:hei + p, p:wid + p, :] = img
    return imgPad

def cleanList(points, nms):
    cleanPoints = [points[0]]
    for i in range(len(points)):
        s = True
        for j in range(len(cleanPoints)):
            dist = ((points[i][0] - cleanPoints[j][0]) ** 2 + (points[i][1] - cleanPoints[j][1]) ** 2) ** (1 / 2)
            if (dist < nms):
                s = False
                if points[i][2] > cleanPoints[j][2]:
                    cleanPoints.remove(cleanPoints[j])
                    s = True
                break
        if (s == True):
            cleanPoints.append(points[i])
    return cleanPoints


def crop(img):
    hei, wid, chan = img.shape
    h, w = (hei - H) // 2, (wid - W) // 2
    return img[h:h + H, w:w + W, :]


def resize(img):
    return cv2.resize(img, (W, H))


def getImages(dir, trn, vld, tst, type, trnN, vldN, tstN):
    allPaths = []
    folders = os.listdir(dir)
    trnLen, vldLen, tstLen = len(trn) + trnN, len(vld) + vldN, len(tst) + tstN
    for folder in folders:
        dirr = dir + '/' + folder
        paths = os.listdir(dirr)
        for path in paths:
            allPaths.append(dirr + '/' + path)
    indexes = np.arange(len(allPaths))
    np.random.shuffle(indexes)
    indexes = indexes[:(trnN + vldN + tstN)]
    print(len(allPaths))
    i = 0
    for index in indexes:
        print(str(i) + ' ' + str(index), allPaths[index])
        img = cv2.imread(allPaths[index])
        if type == 'crop':
            img = crop(img)
        if type == 'resize':
            img = resize(img)

        if len(trn) < trnLen:
            trn.append(img)
        elif len(vld) < vldLen:
            vld.append(img)
        elif len(tst) < tstLen:
            tst.append(img)
        else:
            break
        i += 1
    return trn, vld, tst


def getFeature(imgs):
    dsts = []
    for img in imgs:
        dst, hogImage = hog(img, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True,
                            multichannel=True)
        dsts.append(dst)
    return dsts


def getLabels(Np, Nn):
    return np.int8(np.hstack((np.ones((Np)), np.zeros((Nn)))))


def show(imgs):
    print('show')
    res = np.zeros((imgs[0].shape[0], 1, 3))
    for img in imgs:
        res = np.hstack((res, img))
    return res


def draw(img, points):
    for point in points:
        i, j, w, h = point[0], point[1], point[2], point[3]
        img = cv2.rectangle(img, (i, j), (i + w, j + h), color=(255, 255, 255), thickness=3)
    return img


def detect(img, h, w):
    points = []
    hei, wid, chan = img.shape
    k = 10
    for i in range(0, wid - w, k):
        for j in range(0, hei - h, k):
            face = img[j:j + h, i:i + w, :]
            face = resize(face)
            feature = getFeature([face])
            predicted = model.predict(feature)
            label = predicted[0]
            if label == 1:
                points.append([i, j, w, h])
            print(i, j, label)
    return points


def findFace(img, k):
    h, w = int(H * k), int(W * k)
    points = detect(img, h, w)
    return img, points


def FaceDetector(img):
    hei,wid,chan=img.shape
    img=pad(img,p)
    scales = [0.6, 0.9, 1.5]
    allPoints = []
    maxScale = 0.6
    for scale in scales:
        img, points = findFace(img, scale)
        if len(points) != 0:
            maxScale = scale
            points = cleanList(points, scale * 90)
        for point in points:
            allPoints.append(point)
    allPoints = cleanList(allPoints, maxScale * 90)
    img = draw(img, allPoints)
    return img[p:p+hei,p:p+wid,:]


p=50
H, W = 140, 100
trnNp, vldNp, tstNp = 400, 100, 100
trnNn, vldNn, tstNn = 1000, 100, 100
trn, vld, tst = [], [], []
trnLbl, vldLbl, tstLbl = getLabels(trnNp, trnNn), getLabels(vldNp, vldNn), getLabels(tstNp, tstNn)
getImages('lfw', trn, vld, tst, 'crop', trnNp, vldNp, tstNp)
getImages('256_ObjectCategories', trn, vld, tst, 'resize', trnNn, vldNn, tstNn)
# cv2.imwrite('test.jpg', show(trn))
print(len(trn), len(vld), len(tst))
trnDst = getFeature(trn)
vldDst = getFeature(vld)
tstDst = getFeature(tst)
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model.fit(trnDst, trnLbl)
predicted = model.predict(tstDst)
print('accuracy =', str(metrics.accuracy_score(tstLbl, predicted) * 100), '%')

metrics.plot_roc_curve(model, tstDst, tstLbl)
plt.savefig('res01.jpg')

metrics.plot_precision_recall_curve(model, tstDst, tstLbl)
plt.savefig('res02.jpg')

ap = metrics.average_precision_score(tstLbl, predicted)
print(ap * 100)

cv2.imwrite('res03.jpg', FaceDetector(cv2.imread('Melli.jpg')))
cv2.imwrite('res04.jpg', FaceDetector(cv2.imread('Persepolis.jpg')))
cv2.imwrite('res05.jpg', FaceDetector(cv2.imread('Esteghlal.jpg')))
