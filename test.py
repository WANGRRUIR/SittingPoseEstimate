import torch.utils.data
from opts import opts
from data.handle.fusion import Fusion
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger
from utils.eval import getPreds
import numpy as np




opt = opts().parse()
train_loader = torch.utils.data.DataLoader(Fusion(opt, 'train'), batch_size = 1)
num_epoch = 1
for epoch in range(num_epoch):
    for i, (input1, target2D, target3D, meta) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input1).float()
        target2D_var = torch.autograd.Variable(target2D).float()
        target3D_var = torch.autograd.Variable(target3D).float()

        print(input1,target2D_var,target3D_var,meta.shape)
        if i==0:
            break








