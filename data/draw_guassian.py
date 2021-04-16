from __future__ import division
import numpy as np

# def gaussian2D(radius, sigma=1):
#     # m, n = [(s - 1.) / 2. for s in shape]
#     m, n = radius
#     y, x = np.ogrid[-m:m + 1, -n:n + 1]

#     gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
#     gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
#     return gauss

# def draw_gaussian(fmap, center, radius, k=1):
#     diameter = 2 * radius + 1
#     gaussian = gaussian2D((radius, radius), sigma=diameter / 6)
#     gaussian = np.Tensor(gaussian)
#     x, y = int(center[0]), int(center[1])
#     height, width = fmap.shape[:2]

#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)

#     masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
#         masked_fmap = np.max(masked_fmap, masked_gaussian * k)
#         fmap[y - top:y + bottom, x - left:x + right] = masked_fmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1

    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6.0)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius1(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def gaussian_radius_centernet(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# for mse loss
# def draw_msra_gaussian(heatmap, center, sigma):
#   tmp_size = sigma * 3
#   mu_x = int(center[0] + 0.5)
#   mu_y = int(center[1] + 0.5)
#   w, h = heatmap.shape[0], heatmap.shape[1]
#   ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
#   br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
#   if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
#     return heatmap
#   size = 2 * tmp_size + 1
#   x = np.arange(0, size, 1, np.float32)
#   y = x[:, np.newaxis]
#   x0 = y0 = size // 2
#   g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#   g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
#   g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
#   img_x = max(0, ul[0]), min(br[0], h)
#   img_y = max(0, ul[1]), min(br[1], w)
#   heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
#     heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
#     g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
#   return heatmap


def radius4(det_size, min_overlap=0.7):
    h, w = det_size
    ra = h * 0.1133
    rb = w * 0.1133
    return ra, rb


def gaussian_radius_1(det_size, min_overlap=0.7):
    w, h = det_size
    # ra = 0.3 * h
    # rb = 0.3 * w
   
    ra = 0.1155 * h
    rb = 0.1155 * w
    return ra, rb


def gaussian2D_1(shape, sigmah=1, sigmaw=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-((x * x) / (2 * sigmah * sigmaw) +
                 (y * y) / (2 * sigmah * sigmah)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian_1(heatmap, center, rh, rw, k=1):
    diameterh = 2 * rh + 1
    diameterw = 2 * rw + 1
    gaussian = gaussian2D_1((diameterh,diameterw),
        sigmah=diameterh / 6,
        sigmaw=diameterw / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, rw), min(width - x, rw + 1)
    top, bottom = min(y, rh), min(height - y, rh + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[rh - top:rh + bottom, rw - left:rw + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
