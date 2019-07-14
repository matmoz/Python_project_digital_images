import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math


def load_image(img_l):
    folder_zdj = 'zdj/' + img_l + '.jpg'
    img = cv2.imread(folder_zdj, 0)  # ladujemy zdj
    return img


def filtred_image(img_f):

    # binaryzujemy obraz zmieniamy na odcien szarosci
    min = 100
    min, bw_image = cv2.threshold(img_f, thresh=min, maxval=255, type=cv2.THRESH_BINARY_INV)

    # zamian kolorow czarny na bialy bialy na czarny
    bw_image = np.invert(bw_image)

    # usuwamy szumy ze zdj
    kernel = np.ones((3, 3), np.uint8)
    img_f = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernel, iterations=5)

    return img_f


def getFigure(labelledImage, objNumber): # zapisujemy pixele nalezace do obiektu
    points = []
    for y in range(labelledImage.shape[0]):
        for x in range(labelledImage.shape[1]):
            if labelledImage[y, x] == objNumber:
                points.append((y, x))
    return points


def cog(points): # liczymy srodek ciezkosci obiektu
    mx = 0
    my = 0
    for (y, x) in points:
        mx = mx + x # zliczamy wartosci x
        my = my + y # zlicamy wartosci y
    mx = mx / len(points) # dzielimy przez sume wszytskich pixeli nalezacych do obiektu
    my = my / len(points)
    return [my, mx]


def computeBB(points):
    s = len(points) # ilosc punktow obiektu
    my, mx = cog(points) # srodek ciezkosci obiektu
    r = 0
    for point in points:
        r = r + distance.euclidean(point, (my, mx)) ** 2
    return s / (math.sqrt(2 * math.pi * r))


def computeFeret(points): # obliczamy wspolczynnik fareta
    px = [x for (y, x) in points]
    py = [y for (y, x) in points]
    fx = max(px) - min(px)
    fy = max(py) - min(py)
    return float(fy) / float(fx)

def sum_pixel_ojects(img_s): # procent pixeli obiektow na zdj
    height, width = img_s.shape
    sum_w = 0
    for row in range(height):
        for column in range(width):
            if img_s[row, column] == 0:
                sum_w += 1
    print("obiekty stanowia %f procent zdjecia " % float((((width * height) - sum_w) * 100)/(height*width)))


def counting_objects(img_c):
    # transformacja odleglosciowa
    dist_transform = cv2.distanceTransform(img_c, cv2.DIST_L2, 5)
    # progowanie zwiekszamy biale obietky
    ret, clear_objects = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    label_objects, nb_labels = ndi.label(clear_objects) # funkcja indeksuje obiekty na podstawie maski #[[0,1,0],
                                                                                                       # [1,1,1],
                                                                                                       # [0,1,0]]
    sizes = np.bincount(label_objects.ravel()) # liczy pixele nalezonce do indexowanych obiektow (tlo, ob1, ob2 ...)
    #mask_sizes = sizes > 20 # obiekty musza byc wieksze od 20 zabezpieczenie przed pojedynczymi pikselami
    #mask_sizes[0] = 0
    #print(mask_sizes)
    plt.imshow(label_objects, cmap='hot')
    plt.show()
    print("Ilosc wszystkich pixeli: " + str(label_objects.size) + " \nRozdzielczosc: " + str(label_objects.shape))
    print("Ilosc obiektow na zdj: " + str(nb_labels))

    for i in range(nb_labels): # przechodzimy po wszystkich obiektach
        pts = getFigure(label_objects, i + 1) # zwracamy punkty nalezace do obiektu w tablicy
        bb = computeBB(pts) # obliczamy wspolczynnik Blair-Bliss
        feret = computeFeret(pts) # obliczamy wspolczynnik Fareta
        print('\nObiekt: ', i+1, '\nSrodek ciezkosci: ', cog(pts),)
        print('Blair-Bliss: ', bb, '\nFeret: ', feret, '\n' + ('='*59))


if "__main__" == __name__:
    img = load_image("06")
    img_f = filtred_image(img)
    #sum_pixel_ojects(img_f) # obliczamy procent jaki zajmuja obiekty na zdjeciu
    counting_objects(img_f)

