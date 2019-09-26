#The Code from: https://www.kaggle.com/ratthachat/aptos-updated-albumentation-meets-grad-cam
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from src.utils import expand_path
from src import config
import math

class MyDataset(Dataset):
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        
        p = self.df.id_code.values[idx]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        # image = resize_image(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = crop_image_from_gray(image)
        # image = circle_crop_v3(image)
        image = resize_image(image)
        # image = Radius_Reduction(image)
        # image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur( image , (0,0) , 10), -4, 128)
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
        

def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

# def crop_image_from_gray(img,tol=7):
#     if img.ndim ==2:
#         mask = img>tol
#         return img[np.ix_(mask.any(1),mask.any(0))]
#     elif img.ndim==3:
#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         mask = gray_img>tol
        
#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
#         if (check_shape == 0): # image is too dark so that we crop out everything,
#             return img # return original image
#         else:
#             img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
#             img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
#             img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
#     #         print(img1.shape,img2.shape,img3.shape)
#             img = np.stack([img1,img2,img3],axis=-1)
#     #         print(img.shape)
#         return img


def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            m = np.ix_(mask.any(1), mask.any(0))
            img1 = img[:, :, 0][m]
            img2 = img[:, :, 1][m]
            img3 = img[:, :, 2][m]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop_v3(img, tol=7):
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    mask = img_gray > tol

    # estimate radius
    h, w = img.shape[:2]
    where = np.where(mask[:, 0])[0]
    h1, h2 = where[0], where[-1]
    where = np.where(mask[0, :])[0]
    w1, w2 = where[0], where[-1]

    radius1 = int(round(math.sqrt(w ** 2 + (h2 - h1) ** 2) / 2.))
    radius2 = int(round(math.sqrt(h ** 2 + (w2 - w1) ** 2) / 2.))
    # print(radius1, radius2)

    wider = min(radius1, radius2) * 2
    top = max(0, (wider - h) // 2)
    bottom = max(0, wider - h - top)
    left = max(0, (wider - w) // 2)
    right = max(0, wider - w - left)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return img

def Radius_Reduction(img,PARAM=96):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1

def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r


def resize_image(im, augmentation=False):
    # Crops, resizes and potentially augments the image to IMAGE_SIZE
    cx, cy, r = info_image(im)
    scaling = config.IMG_SIZE/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - config.IMG_SIZE/2
    M[1,2] -= cy - config.IMG_SIZE/2
    return cv2.warpAffine(im,M,(config.IMG_SIZE, config.IMG_SIZE)) # This is the mos


def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 128)