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
    int match_surf_points(np.float32_t *features0, np.int32_t *x0, np.int32_t *y0, np.int32_t *scale0, np.float32_t *orientation0,
                          np.uint8_t *sign0, np.float32_t *cornerness0, int num_points0,
                          np.float32_t *features1, np.int32_t *x1, np.int32_t *y1, np.int32_t *scale1, np.float32_t *orientation1,
                          np.uint8_t *sign1, np.float32_t *cornerness1, int num_points1,
                          int is64, float threshNNRD, float threshNND,
                          np.int32_t **out_matches)

cdef extern from "stdlib.h":
    void free(void *ptr)

cdef extern from "string.h":
    void *memcpy(void *dest, void *src, int n)

cdef class SURF(impoint.BaseFeaturePoint):
    cdef int _max_points
    cdef np.ndarray points, x, y, scale, orientation, sign, cornerness
    cdef float _thresh_nnrd, _thresh_nnd

    def __init__(self, max_points=10000):
        super(SURF, self).__init__()
        self._max_points = max_points
        self._thresh_nnrd = 0.63668169929614182
        self._thresh_nnd = 9303.7827619662221
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
            out.append({'descriptor': self.points[i].copy(),
                        'x': self.x[i],
                        'y': self.y[i],
                        'scale': self.scale[i],
                        'orientation': self.orientation[i],
                        'sign': self.sign[i],
                        'cornerness': self.cornerness[i]})
        return out

    def _convert_points(self, points):
        num_points = len(points)
        descriptor = np.ascontiguousarray(np.zeros((num_points, 64), dtype=np.float32))
        x = np.ascontiguousarray(np.zeros(num_points, dtype=np.int32))
        y = np.ascontiguousarray(np.zeros(num_points, dtype=np.int32))
        scale = np.ascontiguousarray(np.zeros(num_points, dtype=np.int32))
        orientation = np.ascontiguousarray(np.zeros(num_points, dtype=np.float32))
        sign = np.ascontiguousarray(np.zeros(num_points, dtype=np.uint8))
        cornerness = np.ascontiguousarray(np.zeros(num_points, dtype=np.float32))
        for i, point in enumerate(points):
            descriptor[i] = point['descriptor']
            x[i] = point['x']
            y[i] = point['y']
            scale[i] = point['scale']
            orientation[i] = point['orientation']
            sign[i] = point['sign']
            cornerness[i] = point['cornerness']
        return descriptor, x, y, scale, orientation, sign, cornerness

    def match(self, points0, points1):
        cdef np.ndarray descriptor0, x0, y0, scale0, orientation0, sign0, cornerness0
        cdef np.ndarray descriptor1, x1, y1, scale1, orientation1, sign1, cornerness1
        cdef np.int32_t *out_matches
        descriptor0, x0, y0, scale0, orientation0, sign0, cornerness0 = self._convert_points(points0)
        descriptor1, x1, y1, scale1, orientation1, sign1, cornerness1 = self._convert_points(points1)
        print('Prematch')
        sz = match_surf_points(<np.float32_t *>descriptor0.data, <np.int32_t *>x0.data, <np.int32_t *>y0.data,
                               <np.int32_t *>scale0.data, <np.float32_t *>orientation0.data, <np.uint8_t *>sign0.data,
                               <np.float32_t *>cornerness0.data, len(points0),
                               <np.float32_t *>descriptor1.data, <np.int32_t *>x1.data, <np.int32_t *>y1.data,
                               <np.int32_t *>scale1.data, <np.float32_t *>orientation1.data, <np.uint8_t *>sign1.data,
                               <np.float32_t *>cornerness1.data, len(points1),
                               1, self._thresh_nnrd, self._thresh_nnd, &out_matches)
        print('Post')
        cdef np.ndarray matches = np.ascontiguousarray(np.zeros((sz, 2), dtype=np.int32))
        memcpy(matches.data, out_matches, sz * 4 * 2)
        free(out_matches)
        return matches
