import cv2
import numpy as np
from skimage.transform import rotate
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


class ChannelShift:
    def __init__(self, intensity, seed=2022):
        self.name = "ChannelShift"
        assert 1 < intensity < 255, "Set the pixel values to be shifted (1, 255)"
        self.intensity = intensity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        height, width, ch = img.shape
        img = img.astype(np.float32)
        for i in range(ch):
            img[:, :, i] += self.rng.integers(self.intensity) * self.rng.choice([1, -1])
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


class Stripes:
    def __init__(self, horizontal, vertical, space, width, intensity):
        self.name = "Stripes"
        self.horizontal = horizontal
        self.vertical = vertical
        self.space = space
        self.width = width
        self.intensity = intensity

    def apply(self, img):
        h, w, c = img.shape
        g_h = int(h / self.width)
        g_w = int(w / self.width)
        mask = np.zeros([g_h, g_w, c])

        if self.horizontal:
            mask[::self.space, :, :] = self.intensity
        if self.vertical:
            mask[:, ::self.space, :] = self.intensity

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = mask.astype(np.float32) + img.astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


class Blurring:
    def __init__(self, kernel=9, randomness=-1, seed=2022):
        self.name = "Blurring"
        if randomness == -1:
            randomness = kernel - 2
        assert 0 < randomness < kernel, "REQUIREMENT: 0 < randomness ({}) < kernel({})".format(randomness, kernel)
        self.kernel = kernel
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        k = self.kernel + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.blur(img.astype(np.float32), ksize=(k, k))
        return img.astype(np.uint8)


class NeedsMoreJPG:
    def __init__(self, percentage, randomness, seed=2022):
        self.name = "NeedsMoreJPG"
        self.percentage = percentage
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        h, w = img.shape[:2]
        p = self.percentage + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.resize(img, (int(w * p / 100), int(h * p / 100)), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img


class SaltNPepper:
    def __init__(self, max_delta, grain_size):
        self.name = "SaltNPepper"
        self.max_delta = max_delta
        self.grain_size = grain_size

    def apply(self, img):
        h, w, c = img.shape
        snp_h = max(int(h / self.grain_size), 3)
        snp_w = max(int(w / self.grain_size), 3)
        snp = np.random.randint(-self.max_delta, self.max_delta, size=[snp_h, snp_w, c])
        snp = cv2.resize(snp, (w, h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.int) + snp
        return np.clip(img, 0, 255).astype(np.uint8)


def apply_noise(img, tar, percentage):
    noise = SaltNPepper(
        max_delta=int(percentage * 256),
        grain_size=np.random.choice([1, 2, 4, 8])
    )
    return noise.apply(img), tar


def apply_blur(img, lab, percentage):
    height, width, ch = img.shape
    k_w = np.max([int(percentage * width / 2), 2])
    k_h = np.max([int(percentage * height / 2), 2])
    img = cv2.blur(img.astype(np.float32), ksize=(k_w, k_h))
    return img.astype(np.uint8), lab


def apply_channel_shift(img):
    noise = ChannelShift(intensity=16)
    return noise.apply(img)


def apply_horizontal_flip(img, lab, percentage):
    img = cv2.flip(img, 1)
    lab = cv2.flip(lab, 1)
    if len(lab.shape) == 2:
        lab = np.expand_dims(lab, axis=2)
    return img, lab


def apply_vertical_flip(img, lab, percentage):
    img = cv2.flip(img, 0)
    lab = cv2.flip(lab, 0)
    if len(lab.shape) == 2:
        lab = np.expand_dims(lab, axis=2)
    return img, lab


def apply_crop(img, lab, percentage):
    height, width, ch = img.shape
    prz_zoom = 0.10
    w_random = int(width * prz_zoom)
    h_random = int(height * prz_zoom)
    if w_random > 0:
        x1_img = np.random.randint(w_random)
        x2_img = width - np.random.randint(w_random)
    else:
        x1_img = 0
        x2_img = width

    if h_random > 0:
        y1_img = np.random.randint(h_random)
        y2_img = height - np.random.randint(h_random)
    else:
        y1_img = 0
        y2_img = height

    img = img[y1_img:y2_img, x1_img:x2_img, :]
    lab = lab[y1_img:y2_img, x1_img:x2_img, :]
    return img, lab


def apply_rotation_90(img, lab, percentage):
    angle = np.random.choice([0, 90, 180, 270])

    if angle == 270:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 0)
        lab = np.transpose(lab, (1, 0, 2))
        lab = cv2.flip(lab, 0)
        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
    elif angle == 180:
        img = cv2.flip(img, -1)
        lab = cv2.flip(lab, -1)
        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
    elif angle == 90:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 1)
        lab = np.transpose(lab, (1, 0, 2))
        lab = cv2.flip(lab, 1)
        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
    elif angle == 0:
        pass

    return img, lab


def apply_tiny_rotation(img, lab, percentage):
    img = img.astype(np.float)
    lab = lab.astype(np.float)
    rand_angle = np.random.randint(20) - 10
    img = rotate(img, angle=rand_angle, mode="reflect")
    lab = rotate(lab, angle=rand_angle, mode="reflect")
    return img.astype(np.uint8), lab.astype(np.int)


def apply_mask(img, lab, percentage):
    height, width, ch = img.shape
    # Recalculate percentage to cover defined percentage of image
    percentage = np.sqrt(100*100*percentage) / 100

    mask_height = int(percentage * height)
    mask_width = int(percentage * width)
    y1_img = np.random.randint(height - mask_height)
    x1_img = np.random.randint(width - mask_width)
    img[y1_img:y1_img + mask_height, x1_img:x1_img + mask_width, 0] = np.random.randint(0, 255)
    img[y1_img:y1_img + mask_height, x1_img:x1_img + mask_width, 1] = np.random.randint(0, 255)
    img[y1_img:y1_img + mask_height, x1_img:x1_img + mask_width, 2] = np.random.randint(0, 255)
    return img, lab


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def norm(data):
    min_mat = np.min(data)
    max_mat = np.max(data)
    return (data - min_mat) / (max_mat - min_mat)


def apply_blackhole_mask(img, lab, percentage):
    height, width, ch = img.shape
    # Recalculate percentage to cover defined percentage of image
    percentage = np.sqrt(100 * 100 * percentage) / 100

    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    mx = np.random.randint(width)
    my = np.random.randint(height)
    sx = int(percentage * width)
    sy = int(percentage * height)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    g_m = gaus2d(x, y, mx, my, sx, sy)
    g_m = np.expand_dims(g_m, axis=2)
    g_m = np.concatenate([g_m, g_m, g_m], axis=2)

    g_m = np.clip(norm(g_m) * 1.5, 0, 1)
    g_m = np.ones(g_m.shape) - g_m
    img = img * g_m

    # img[y1_img:y1_img + mask_height, x1_img:x1_img + mask_width, :] = masked_area
    return img, lab


def apply_to_edge_image(img, tar, percentage):
    height, width, ch = img.shape
    k = 7
    for c in range(3):
        img[:, :, c] = 255 * norm(cv2.Laplacian(img[:, :, c], -1, ksize=k))
    return img, tar


def apply_cross_cut(img, lab, percentage):
    height, width, ch = img.shape
    # Recalculate percentage to cover defined percentage of image
    percentage = (100 - np.sqrt(100*100*(1 - percentage))) / 100

    cross_height = int(percentage * height)
    cross_width = int(percentage * width)
    y1_img = np.random.randint(height - cross_height)
    x1_img = np.random.randint(width - cross_width)

    top_left = img[0:y1_img, 0:x1_img, :]
    top_right = img[0:y1_img, x1_img+cross_width:, :]
    bot_left = img[y1_img+cross_height:, 0:x1_img, :]
    bot_right = img[y1_img+cross_height:, x1_img+cross_width:, :]

    top = np.concatenate([top_left, top_right], axis=1)
    bot = np.concatenate([bot_left, bot_right], axis=1)
    img = np.concatenate([top, bot], axis=0)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img, lab


def img_to_patches(img, window_size):
    tiles = []
    for i in range(0, img.shape[0], window_size):
        for j in range(0, img.shape[1], window_size):
            tiles.append(img[i:i + window_size, j:j + window_size, :])
    return tiles


def patches_to_img(patches, img_size):
    img = np.zeros((img_size, img_size, 3))
    window_size = patches[0].shape[0]
    c = 0
    for i in range(0, img.shape[0], window_size):
        for j in range(0, img.shape[1], window_size):
            img[i:i + window_size, j:j + window_size, :] = patches[c]
            c += 1
    return img


def rotate_patch(patch):
    angle = np.random.choice([90, 180, 270])
    if angle == 270:
        patch = np.transpose(patch, (1, 0, 2))
        patch = cv2.flip(patch, 0)
    elif angle == 180:
        patch = cv2.flip(patch, -1)
    else:
        patch = np.transpose(patch, (1, 0, 2))
        patch = cv2.flip(patch, 1)
    return patch


def apply_patch_rotation(img, lab, n_patches=8, percentage=0.25):
    height, width, ch = img.shape
    size = 128
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    patches = img_to_patches(img, int(size / n_patches))

    patch_selected = np.arange(len(patches))
    patch_selected = np.random.choice(patch_selected, int(percentage * len(patches)), replace=False)

    for p_i, patch in enumerate(patches):
        if p_i in patch_selected:
            patch = rotate_patch(patch)
        patches[p_i] = patch
    img = patches_to_img(patches, size)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img, lab


def apply_patch_shuffling(img, lab, n_patches=8, percentage=0.25):
    height, width, ch = img.shape
    size = 128
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    patches = img_to_patches(img, int(size / n_patches))

    patch_selected = np.arange(len(patches))
    patch_selected = np.random.choice(patch_selected, int(percentage * len(patches)), replace=False)
    patch_shuffled = np.copy(patch_selected)
    np.random.shuffle(patch_shuffled)

    patch_mapping = {i: j for i, j in zip(patch_selected, patch_shuffled)}

    for p_i, patch in enumerate(patches):
        if p_i in patch_selected:
            patch = patches[patch_mapping[p_i]]
        patches[p_i] = patch
    img = patches_to_img(patches, size)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img, lab


class Augmentations:
    def __init__(self,
                 neutral=1.0,
                 masking=0.0,
                 cross_cut=0.0,
                 patch_rotation=0.0,
                 patch_shuffling=0.0,
                 black_hole=0.0,
                 blurring=0.0,
                 noise=0.0,
                 flip_rotate90=1.0,
                 crop=1.0):

        self.tasks = [
            {"name": "NEUTRAL", "function": dummy_function, "percentage": neutral},
            {"name": "MASKING", "function": apply_mask, "percentage": masking},
            {"name": "CROSS_REMOVAL", "function": apply_cross_cut, "percentage": cross_cut},
            {"name": "PATCH_ROTATION", "function": apply_patch_rotation, "percentage": patch_rotation},
            {"name": "PATCH_SHUFFLING", "function": apply_patch_shuffling, "percentage": patch_shuffling},
            {"name": "BLACKHOLE_MASKING", "function": apply_blackhole_mask, "percentage": black_hole},
            {"name": "BLURRING", "function": apply_blur, "percentage": blurring},
            {"name": "NOISE", "function": apply_noise, "percentage": noise},
            {"name": "ROTATION", "function": apply_rotation_90, "percentage": flip_rotate90},
            {"name": "VERTICAL_FLIP", "function": apply_vertical_flip, "percentage": flip_rotate90},
            {"name": "HORIZONTAL_FLIP", "function": apply_horizontal_flip, "percentage": flip_rotate90},
            {"name": "CROP", "function": apply_crop, "percentage": crop},
        ]

        self.active_tasks = [t for t in self.tasks if t["percentage"] > 0.0]

    def apply(self, img, tar):
        if len(self.active_tasks) == 0:
            return img, tar
        task_to_apply = np.random.choice(self.active_tasks)
        percentage = np.random.randint(100 * task_to_apply["percentage"]) / 100
        if percentage == 0:
            return img, tar
        return task_to_apply["function"](img, tar, percentage=percentage)


def dummy_function(img, tar, percentage):
    return img, tar


class EncoderTask:
    def __init__(self,
                 neutral=0.0,
                 masking=0.0,
                 cross_cut=0.0,
                 patch_rotation=0.0,
                 patch_shuffling=0.0,
                 black_hole=0.0,
                 blurring=0.0,
                 noise=0.0,
                 flip_rotate90=0.0):

        self.tasks = [
            {"name": "NEUTRAL", "function": dummy_function, "percentage": neutral},
            {"name": "MASKING", "function": apply_mask, "percentage": masking},
            {"name": "CROSS_REMOVAL", "function": apply_cross_cut, "percentage": cross_cut},
            {"name": "PATCH_ROTATION", "function": apply_patch_rotation, "percentage": patch_rotation},
            {"name": "PATCH_SHUFFLING", "function": apply_patch_shuffling, "percentage": patch_shuffling},
            {"name": "BLACKHOLE_MASKING", "function": apply_blackhole_mask, "percentage": black_hole},
            {"name": "BLURRING", "function": apply_blur, "percentage": blurring},
            {"name": "NOISE", "function": apply_noise, "percentage": noise},
            {"name": "ROTATION", "function": apply_rotation_90, "percentage": flip_rotate90},
            {"name": "VERTICAL_FLIP", "function": apply_vertical_flip, "percentage": flip_rotate90},
            {"name": "HORIZONTAL_FLIP", "function": apply_horizontal_flip, "percentage": flip_rotate90},
        ]

        self.active_tasks = [t for t in self.tasks if t["percentage"] > 0.0]

    def apply(self, img, tar):
        if len(self.active_tasks) == 0:
            return img, tar
        task_to_apply = np.random.choice(self.active_tasks)
        return task_to_apply["function"](img, tar, percentage=task_to_apply["percentage"])


def tests():
    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_mask(img, img, percentage=0.25)
    cv2.imwrite("./test_image/test_traffic_sign_masked.png", img)

    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_cross_cut(img, img, percentage=0.25)
    cv2.imwrite("./test_image/test_traffic_sign_cross_cut.png", img)

    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_patch_rotation(img, img)
    cv2.imwrite("./test_image/test_traffic_sign_patch_rotation.png", img)

    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_patch_shuffling(img, img)
    cv2.imwrite("./test_image/test_traffic_sign_patch_shuffle.png", img)

    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_blackhole_mask(img, img, percentage=0.25)
    cv2.imwrite("./test_image/test_traffic_sign_masked_clackhole.png", img)

    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_blur(img, img, percentage=0.75)
    cv2.imwrite("./test_image/test_traffic_sign_blur.png", img)

    img = cv2.imread("./test_image/test_traffic_sign.png")
    img, _ = apply_noise(img, img, percentage=0.90)
    cv2.imwrite("./test_image/test_traffic_sign_noise.png", img)


if __name__ == "__main__":
    tests()
