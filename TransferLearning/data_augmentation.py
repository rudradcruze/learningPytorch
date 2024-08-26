# Imports
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import CatsAndDogsDataset

# Load Data
my_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5,
                           saturation=0.5, hue=0.5),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv',
                             root_dir='cats_dogs',
                             transform=my_transforms)

image_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img' + str(image_num) + '.png')
        image_num += 1