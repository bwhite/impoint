import cv
import numpy as np
cimport numpy as np
cimport impoint
import imfeat

cdef extern from "surf_feature.hpp":
    int compute_surf_descriptors(np.uint8_t *data, int height, int width, int max_points, float *points)
    int compute_surf_points(np.uint8_t *data, int height, int width, int max_points, np.float32_t *points,
                            np.int32_t *x, np.int32_t *y, np.int32_t *scale, np.float32_t *orientation,
                            np.uint8_t *sign, np.float32_t *cornerness)
    int compute_surf_random(np.uint8_t *data, int height, int width, int max_points, np.float32_t *points)

cdef class SURF(impoint.BaseFeaturePoint):
    cdef int _max_points
    cdef np.ndarray points, x, y, scale, orientation, sign, cornerness

    def __init__(self, max_points=10000):
        super(SURF, self).__init__()
        self._max_points = max_points
        # Allocate memory for all of the things we need
        self.points = np.ascontiguousarray(np.zeros((max_points, 64), dtype=np.float32))
        self.x = np.ascontiguousarray(np.zeros(max_points, dtype=np.int32))
        self.y = np.ascontiguousarray(np.zeros(max_points, dtype=np.int32))
        self.scale = np.ascontiguousarray(np.zeros(max_points, dtype=np.int32))
        self.orientation = np.ascontiguousarray(np.zeros(max_points, dtype=np.float32))
        self.sign = np.ascontiguousarray(np.zeros(max_points, dtype=np.uint8))
        self.cornerness = np.ascontiguousarray(np.zeros(max_points, dtype=np.float32))


    def __call__(self, image_in):
        image_in = imfeat.convert_image(image_in, [('opencv', 'gray', 8)])
        cdef np.ndarray image = np.ascontiguousarray(cv.GetMat(image_in), dtype=np.uint8)
        cdef int height = image.shape[0]
        cdef int width = image.shape[1]
        max_points = min(self._max_points, height * width)
        num_pts = compute_surf_points(<np.uint8_t *>image.data,
                                      height,
                                      width,
                                      max_points,
                                      <np.float32_t *>self.points.data,
                                      <np.int32_t *>self.x.data,
                                      <np.int32_t *>self.y.data,
                                      <np.int32_t *>self.scale.data,
                                      <np.float32_t *>self.orientation.data,
                                      <np.uint8_t *>self.sign.data,
                                      <np.float32_t *>self.cornerness.data)
        # Construct output
        out = []
        for i in range(num_pts):
            out.append({'x': self.x[i],
                        'y': self.y[i],
                        'scale': self.scale[i],
                        'orientation': self.orientation[i],
                        'sign': self.sign[i],
                        'cornerness': self.cornerness[i]})
        return out
