import impoint
import Image
import cv
import random


def plot_points(image, points, color=(255, 0, 0)):
    for point in points:
        cv.Circle(image, (int(point['x']), int(point['y'])), radius=5, color=color)


def plot_matches(image, matches, points0, points1, color=(0, 255, 0)):
    for match in matches:
        point0 = points0[match[0]]
        point1 = points1[match[1]]
        cv.Line(image, (int(point0['x']), int(point0['y'])), (int(point1['x']), int(point1['y'])), color=color)


image0 = cv.LoadImage('frame0.png')
image1 = cv.LoadImage('frame1.png')
surf = impoint.SURF()
points0 = surf(image0)
points1 = surf(image1)
plot_points(image0, points0, (255, 0, 0))
plot_points(image0, points1, (0, 0, 255))
matches = surf.match(points0, points1)
plot_matches(image0, matches, points0, points1)
print('P0[%d] P1[%d] Matches[%d]' % (len(points0), len(points1), len(matches)))

cv.SaveImage('match_out.jpg', image0)
