import impoint
import Image
import cv
import random
import numpy as np


def plot_points(image, points, color=(255, 0, 0)):
    for point in points:
        cv.Circle(image, (int(point['x']), int(point['y'])), radius=5, color=color)


def plot_matches(image, matches, points0, points1, color=(0, 255, 0)):
    for match in matches:
        point0 = points0[match[0]]
        point1 = points1[match[1]]
        cv.Line(image, (int(point0['x']), int(point0['y'])), (int(point1['x']), int(point1['y'])), color=color)


def points_to_cvmat(points):
    a = np.asfarray([(point['x'], point['y']) for point in points])
    m = cv.CreateMatHeader(len(points), 2, cv.CV_64F)
    cv.SetData(m, a.tostring())
    return m


def match_points(matches, points0, points1):
    return zip(*[(points0[x], points1[y]) for x, y in matches])


image0 = cv.LoadImage('frame0.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
image1 = cv.LoadImage('frame1.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
surf = impoint.SURF()
points0 = surf(image0)
points1 = surf(image1)
matches = surf.match(points0, points1)
m0, m1 = map(points_to_cvmat, match_points(matches, points0, points1))
h = cv.CreateMat(3, 3, cv.CV_64F)
cv.FindHomography(m0, m1, h, method=cv.CV_RANSAC, ransacReprojThreshold=10)
print(np.asarray(h))
warped = cv.CreateImage((image0.width, image0.height), 8, 1)
cv.WarpPerspective(image0, warped, h)
cv.SaveImage('warped.png', np.array(np.abs(np.asfarray(cv.GetMat(warped)) + np.asfarray(cv.GetMat(image1)))/2, dtype=np.uint8))
