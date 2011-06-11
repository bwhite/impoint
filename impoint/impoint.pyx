import warnings


cdef class BaseFeaturePoint(object):

    def __init__(self):
        super(BaseFeaturePoint, self).__init__()

    def __call__(self, image):
        raise NotImplementedError

    def describe(self, points):
        raise NotImplementedError

    def match(self, points):
        raise NotImplementedError
