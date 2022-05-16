# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:03:49 2022

@author: 95118
"""

import torch.nn as nn
from torchsummary import summary
import torchvision.models as models

model = models.__dict__['resnet18']()#导入模型
input_num = model.fc.in_features
model.fc = nn.Linear(input_num,200)#将 output 修改为 200 维。

#打印模型和参数，输入图片大小为3*64*64
summary(model, (3, 64, 64),device="cpu")#默认是gpu，所以要专门加上“device=‘cpu’”