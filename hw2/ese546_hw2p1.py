from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Show the result of augmenting this image using the following 10 augmentations:
# (a) ShearX, (b) ShearY, (c) TranslateX, (d) TranslateY, (e) Rotate, (d) Brightness,
# (e) Color, (f) Contrast, (g) Sharpness, (h) Posterize, (i) Solarize and (j) Equalize

torch.manual_seed(0)

original_img = Image.open(Path('vlcsnap-2024-07-07-22h36m07s993.png')).convert('RGB')

# Define the transformations
transformations = {
    'ShearX: degrees=0, shear=(10, 10)': transforms.RandomAffine(degrees=0, shear=(10, 10)),
    'ShearY: degrees=0, shear=(0, 0, 10, 10)': transforms.RandomAffine(degrees=0, shear=(0, 0, 10, 10)),
    'TranslateX: degrees=20, translate=(0.3, 0)': transforms.RandomAffine(degrees=20, translate=(0.3, 0)),
    'TranslateY: degrees=5, translate=(0, 0.2)': transforms.RandomAffine(degrees=5, translate=(0, 0.2)),
    'Rotate: degrees=45': transforms.RandomRotation(degrees=45),
    'Brightness: brightness=0.5': transforms.ColorJitter(brightness=0.5),
    'Color: saturation=2.0': transforms.ColorJitter(saturation=2.0),
    'Contrast: contrast=0.8': transforms.ColorJitter(contrast=0.8),
    'Sharpness: sharpness_factor=5.0': transforms.RandomAdjustSharpness(sharpness_factor=5.0),
    'Posterize: p=0.9, bits=1': transforms.RandomPosterize(p=0.9, bits=1),
    'Solarize: threshold=128': transforms.RandomSolarize(threshold=128),
    'Equalize: p=0.5': transforms.RandomEqualize()
}

# Apply each transformation and display
fig, ax = plt.subplots(4, 3, figsize=(12, 16))
for idx, (key, transform) in enumerate(transformations.items()):
    transformed_img = transform(original_img)
    ax[idx // 3, idx % 3].imshow(transformed_img)
    ax[idx // 3, idx % 3].title.set_text(key)
    ax[idx // 3, idx % 3].axis('off')

plt.tight_layout()
plt.show()