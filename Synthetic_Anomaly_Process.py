import random
import math
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import re
from torch.utils.data import Dataset
from einops import rearrange
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
def set_random_seed(seed=1):
    np.random.seed(1)
    seed = 1
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # enable cudnn backend
    os.environ['PYTHONHASHSEED'] = '0'
    torch.backends.cudnn.enabled = True
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(1)

img = Image.open("C:/Users/MVCLAB/Desktop/tools/pcb4/0047.JPG")
resize_image = transforms.Resize([448,448])
img = resize_image(img)
image_np = np.array(img)
img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

_, mask_target_background = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
mask_target_background = mask_target_background.astype(bool).astype(int)
mask_target_foreground = -(mask_target_background - 1)

augmenters = [
    iaa.GammaContrast((0.5,2.0),per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.AddToHueAndSaturation((-50,50),per_channel=True),
    iaa.Solarize(0.5, threshold=(32,128)),
    iaa.Posterize(),
    iaa.Invert(),
    iaa.pillike.Autocontrast(),
    iaa.pillike.Equalize(),
    iaa.Affine(rotate=(-45, 45))
]

aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
aug = iaa.Sequential([
    augmenters[aug_ind[0]],
    augmenters[1],
    augmenters[aug_ind[2]]
])
print(image_np.shape)
structure_source_img = aug(image = image_np)
plt.imshow(structure_source_img)
plt.show()

structure_source_img = rearrange(
    tensor  = structure_source_img, 
    pattern = '(h gh) (w gw) c -> (h w) gw gh c',
    gw      = 56, 
    gh      = 56
)
disordered_idx = np.arange(structure_source_img.shape[0])
np.random.shuffle(disordered_idx)

structure_source_img = rearrange(
    tensor  = structure_source_img[disordered_idx], 
    pattern = '(h w) gw gh c -> (h gh) (w gw) c',
    h       = 8,
    w       = 8
).astype(np.float32)

plt.imshow(structure_source_img.astype(np.uint8))
plt.show()

fig, ax = plt.subplots(1,2, figsize=(10,15))
ax[0].imshow(mask_target_background, cmap='gray')
ax[0].set_title('Background')
ax[1].imshow(mask_target_foreground, cmap='gray')
ax[1].set_title('Foreground')
plt.show()

print(structure_source_img.shape)
factor = np.random.uniform(
    0.15, 
    1, 
    size=1
)[0]

print('factor: ',factor)

from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)

perlin_scale = 6
min_perlin_scale = 0
perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

noise = generate_perlin_noise_2d((448, 448), (perlin_scalex, perlin_scaley))
plt.imshow(noise, cmap='gray', interpolation='lanczos')
plt.show() 

rot = iaa.Affine(rotate=(45, 50))
perlin_noise = rot(image=noise)
plt.imshow(perlin_noise, cmap='gray')
plt.show()

print('max: ',perlin_noise.max())
print('min: ',perlin_noise.min())

threshold = 0.40 # 調整裂痕大小 
mask_noise = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
plt.imshow(mask_noise, cmap='gray')
plt.show()

mask = mask_noise * mask_target_background
mask = np.expand_dims(mask, axis=2)
plt.imshow(mask, cmap='gray')
plt.show()

texture_source_img = cv2.imread("D:/Downloads/dtd/images/banded/banded_0077.jpg")
texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
texture_source_img = cv2.resize(
    texture_source_img, 
    dsize=(448, 448)
).astype(np.float32)
plt.imshow(texture_source_img.astype(np.uint8))
plt.show()

texture_source_img = factor * (mask * texture_source_img) + (1 - factor) * (mask * img)
structure_source_img = factor * (mask * structure_source_img) + (1 - factor) * (mask * img)
fig, ax = plt.subplots(1,2, figsize=(10,15))
ax[0].imshow(texture_source_img.astype(np.uint8))
ax[0].set_title('texture')
ax[1].imshow(structure_source_img.astype(np.uint8))
ax[1].set_title('structure')
plt.show()

texture_anomaly = ((- mask + 1) * img) + texture_source_img
structure_anomaly = ((- mask + 1) * img) + structure_source_img
fig, ax = plt.subplots(1,2, figsize=(10,15))
ax[0].imshow(texture_anomaly.astype(np.uint8))
ax[0].set_title('texture anomaly')
ax[1].imshow(structure_anomaly.astype(np.uint8))
ax[1].set_title('structure anomaly')

print(type(texture_anomaly))
print(type(structure_anomaly))
print("------------------------------------------------------")
texture_image = Image.fromarray(texture_anomaly, mode = 'RGB')
structure_image = Image.fromarray(structure_anomaly, mode = 'RGB')
print(type(texture_image))
print(type(structure_image))
