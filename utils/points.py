import impoint
import Image
import cv
import random


image = cv.LoadImageM('lena.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
points = impoint.SURF()(image)
points_copy = list(points)
random.shuffle(points_copy)

points2 = impoint.SURFCV()(image)
out = impoint.SURF().match(points, points_copy)
correct = 0
for i, j in out:
    correct += int(points[i] == points_copy[j])
print(correct / float(len(out)))

for point in points:
    cv.Circle(image, (int(point['x']), int(point['y'])), radius=5, color=(255, 0, 0))
#Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
cv.SaveImage('out.jpg', image)
