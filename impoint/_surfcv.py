import imfeat
import impoint
import cv


class SURFCV(impoint.BaseFeaturePoint):
    def __init__(self):
        super(SURFCV, self).__init__()

    def __call__(self, image):
        image = imfeat.convert_image(image, [('opencv', 'gray', 8)])
        keypoints, descriptors = cv.ExtractSURF(image, None, cv.CreateMemStorage(), (0, 500, 3, 4))
        out = []
        for ((x, y), laplacian, size, direction, hessian) in keypoints:
            out.append({'x': x,
                        'y': y,
                        'scale': size,
                        'orientation': direction,
                        'sign': laplacian,
                        'cornerness': hessian})
        return out
