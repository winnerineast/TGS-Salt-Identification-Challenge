from os import listdir
from os.path import isfile, join
import PIL.Image as image
import numpy as np
from sklearn.cluster import KMeans


def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data), m, n


picture_files = [f for f in listdir('../input/test/images') if isfile(join('../input/test/images', f))]

for file_name in picture_files:
    print('Load ' + '../input/test/images/' + file_name)
    imgData, row, col = loadData('../input/test/images/' + file_name)
    label = KMeans(n_clusters=2).fit_predict(imgData)
    label = label.reshape([row, col])
    pic_new = image.new("L", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
    print('Save ' + '../input/test/masks/' + file_name)
    pic_new.save('../input/test/masks/' + file_name, "PNG")
