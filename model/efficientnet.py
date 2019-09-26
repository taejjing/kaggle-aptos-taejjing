from efficientnet_pytorch import EfficientNet
from src import config
import torch

def get_efn_model(model_num, isPretrained=True):

    if isPretrained :
        print("Pretrained Weight loaded")
        model = EfficientNet.from_pretrained('efficientnet-b{}'.format(model_num))
    else :
        print("No Pretrained Weight")
        model = EfficientNet.from_name('efficientnet-b{}'.format(model_num))

    return model