import impoint
import Image
import cv

image = cv.LoadImageM('lena.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
points = impoint.SURF()(image)
points2 = impoint.SURFCV()(image)
for point in points:
    cv.Circle(image, (int(point['x']), int(point['y'])), radius=5, color=(255, 0, 0))
#Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
cv.SaveImage('out.jpg', image)
