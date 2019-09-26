from src.datautils import *
from torchvision import transforms
import pandas as pd
import torch
import os
from src import config
import cv2
import time

# train_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation((-120, 120)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'train.csv'))

# trainset     = MyDataset(train_csv, transform=train_transform)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4)

train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'train.csv'))
f_names = train_csv['id_code']

def preprocessing_v1(f_names):

    for f in f_names :
        start_time = time.time()

        f_name = f + '.png'
        a_img_path = os.path.join(config.TRAIN_PATH, f_name)

        image = cv2.imread(a_img_path)
        # cv2.imwrite('tmp/%s_orig.png' % f_name, image)

        image = crop_image_from_gray(image)
        image = resize_image(image)
        # cv2.imwrite('tmp/%s_m_resize.png' % f_name, image)

        # image = circle_crop_v3(image)
        # image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur( image , (0,0) , 10), -4, 128)

        # image = subtract_median_bg_image(image)
        # cv2.imwrite('tmp/%s_sub_med.png' % f_name, image)
        
        # image = Radius_Reduction(image)

        folder_name = 'tmp/crop_gray_resize'
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        cv2.imwrite(f'{folder_name}/{f_name}', image)

        finish_time = time.time() - start_time

        print("Preprocess time : %s" % finish_time)

def preprocessing_v2(f_names):

    for f in f_names :
        start_time = time.time()

        f_name = f + '.png'
        a_img_path = os.path.join(config.TRAIN_PATH, f_name)

        image = cv2.imread(a_img_path)
        # cv2.imwrite('tmp/%s_orig.png' % f_name, image)

        image = resize_image(image)
        # cv2.imwrite('tmp/%s_m_resize.png' % f_name, image)

        # image = crop_image_from_gray(image)
        # image = circle_crop_v3(image)
        # cv2.imwrite('tmp/%s_v3_crop.png' % f_name, image)

        image = subtract_median_bg_image(image)
        # cv2.imwrite('tmp/%s_sub_med.png' % f_name, image)

        # lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl,a,b))
        # image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # cv2.imwrite('tmp/%s_clahe.png' % f_name, image)

        # image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur( image , (0,0) , 30), -4, 128)
        # cv2.imwrite('tmp/%s_resize_ben.png' % f_name, image)
        # cv2.imwrite('tmp/%s_resize.png' % f_name, image)

        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur( image , (0,0) , 30), -4, 128)
        
        image = Radius_Reduction(image)
        cv2.imwrite('tmp/%s_resize_rad_reduc.png' % f_name, image)

        finish_time = time.time() - start_time

        print("Preprocess time : %s" % finish_time)

preprocessing_v1(f_names[:10])