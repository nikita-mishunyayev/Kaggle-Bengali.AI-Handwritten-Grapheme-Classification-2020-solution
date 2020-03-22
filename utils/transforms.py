import random
import cv2
import numpy as np
from albumentations import ImageOnlyTransform, CoarseDropout
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F


class GridMask(DualTransform):
    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                                int(i * grid_h) : int(i * grid_h + grid_h / 2),
                                int(j * grid_w) : int(j * grid_w + grid_w / 2)
                                ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                    int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                    int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                                    ] = self.fill_value

                            if self.mode == 1:
                                this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')

class RandomMorph(ImageOnlyTransform):
    def __init__(self, _min=2, _max=6, element_shape=cv2.MORPH_ELLIPSE, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self._min = _min
        self._max = _max
        self.element_shape = element_shape

    def apply(self, image, **params):
        arr = np.random.randint(self._min, self._max, 2)
        kernel = cv2.getStructuringElement(self.element_shape, tuple(arr))

        _h, _w, c = image.shape
        is_grayscale = c == 1

        if random.random() > 0.5:
            image = cv2.erode(image, kernel, iterations=1)
        else:
            image = cv2.dilate(image, kernel, iterations=1)

        if is_grayscale:
            return np.expand_dims(image, -1)
        else:
            return image

class RandomProjective(ImageOnlyTransform):
    def __init__(self, magnitude=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.magnitude = np.random.uniform(-1, 1) * 0.5 * magnitude

    def apply(self, image, **params):
        height, width, channel = image.shape
        is_grayscale = channel == 1

        x0, y0 = 0, 0
        x1, y1 = 1, 0
        x2, y2 = 1, 1
        x3, y3 = 0, 1

        mode = np.random.choice(['top','bottom','left','right'])

        if mode =='top':
            x0 += self.magnitude
            x1 -= self.magnitude
        if mode =='bottom':
            x3 += self.magnitude
            x2 -= self.magnitude
        if mode =='left':
            y0 += self.magnitude
            y3 -= self.magnitude
        if mode =='right':
            y1 += self.magnitude
            y2 -= self.magnitude

        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * [[width, height]]
        d = np.array([[x0,y0], [x1,y1], [x2,y2], [x3,y3]]) * [[width, height]]

        transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))

        image = cv2.warpPerspective(
                image,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)

        if is_grayscale:
            return np.expand_dims(image, -1)
        else:
            return image


class RandomPerspective(ImageOnlyTransform):
    def __init__(self, magnitude=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.magnitude = np.random.uniform(-1, 1, (4,2)) * 0.25 * magnitude

    def apply(self, image, **params):
        height, width, channel = image.shape
        is_grayscale = channel == 1

        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        d = s + self.magnitude
        s *= [[width, height]]
        d *= [[width, height]]

        transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))

        image = cv2.warpPerspective(
                image,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)

        if is_grayscale:
            return np.expand_dims(image, -1)
        else:
            return image

class NormalizeToMax(ImageOnlyTransform):
    def apply(self, image, **params):
        return (image * (255.0 / image.max())).astype(np.uint8)
