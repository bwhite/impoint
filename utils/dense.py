import impoint
import cv2
import numpy as np
import itertools
import imfeat
a = cv2.imread('lena.jpg')
b = impoint.SURF()
s = 7
bounds = b.compute_dense_bounds(a.shape[0], a.shape[1], s)
#17*s+2
#c = list(((x, y, s) for x, y in itertools.product(np.arange(bounds['y'][0], bounds['y'][1], 10), np.arange(bounds['x'][0], bounds['x'][1], 10))))

print(bounds)
print(b.compute_dense(a).shape)

# Compute clusters and output
clusters = imfeat.BoVW.cluster([a], b.compute_dense, 8)
for x in [1, 2, 4, 8]:
    cv2.imwrite('out_mask-%d.png' % x, np.array(b.make_feature_mask(a, clusters, scale=x), dtype=np.uint8) * 32)
