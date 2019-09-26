from torchvision import transforms
from src import config
import PIL

train_transform_v1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_transform_v2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# train_transform_v2 = transforms.Compose([
#     transforms.CenterCrop(config.IMG_SIZE),        
#     transforms.RandomApply([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
#         transforms.RandomAffine(0, translate=(
#             0.2, 0.2), resample=PIL.Image.BICUBIC),
#         transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),
#         transforms.RandomAffine(0, scale=(0.8, 1.2),
#                                 resample=PIL.Image.BICUBIC)
#     ],p=0.85)
# ])