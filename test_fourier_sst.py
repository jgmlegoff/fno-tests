"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import argparse
import pickle
import math
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from Adam import Adam

from skimage.transform import resize

torch.manual_seed(0)
np.random.seed(0)

def resize_fn(img, size):
    return T.Resize(size, T.InterpolationMode.BICUBIC)(img)



################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################

ntrain = 366
ntest = 122

batch_size = 6
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 5
h = int(((421 - 1)/r) + 1)
#s = h

def predict(x,y,y_normalizer):
    myloss = LpLoss(size_average=False)
    with torch.no_grad():
        test_l2 = 0.0
        x, y = x.cuda(), y.cuda()

        out = model(x).reshape(1, x.shape[1], x.shape[2])
        out = y_normalizer.decode(out)

        test_l2 += myloss(out.view(1,-1), y.view(1,-1)).item()

    y = y.cpu()
    out = out.cpu()
    
    return(y,out, test_l2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--pickles', action=argparse.BooleanOptionalAction)
    parser.add_argument('--superres', action=argparse.BooleanOptionalAction)
    parser.add_argument('--res', action=argparse.BooleanOptionalAction)
    parser.add_argument('--size', default=16)
    parser.add_argument('--model', default=None)
    parser.add_argument('--mrmse', action=argparse.BooleanOptionalAction)
    parser.add_argument('--plotfour', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Size of the input
    s = int(args.size)

    ################################################################
    # load data and data normalization                              
    ################################################################

    TEST_PATH =  '/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz'

### Test data
    if not args.pickles : 
        print('Loading test data')
        npz_data = np.load(TEST_PATH)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar'])
        ssh_data_test = npz_data["FdataAllVar"][0,:,:,0:n_lat]
        sst_data_test = npz_data["FdataAllVar"][1,:,:,0:n_lat]
        with open('ssh_pickle_test.bin', 'wb') as f1:
            pickle.dump(ssh_data_test, f1)
        with open('sst_pickle_test.bin', 'wb') as f2:
            pickle.dump(sst_data_test, f2)

        print(n_var, n_time, n_lat, n_lon)
    
    else : 
        print('Loading test data from pickles')
        with open('ssh_pickle_test.bin', "rb") as f:
            ssh_data_test = pickle.load(f)
        with open('sst_pickle_test.bin', "rb") as f: 
            sst_data_test = pickle.load(f)
        print('Data loaded')


    ssh_data_test_lr = resize_fn(T.ToTensor()(ssh_data_test).permute(1,2,0), (s,s))
    sst_data_test_lr = resize_fn(T.ToTensor()(sst_data_test).permute(1,2,0), (s,s))

    if args.plot:
        fig, axs = plt.subplots(2, 2,figsize=(18, 18))
        pos0 = axs[0,0].imshow(ssh_data_test_lr[0][...,np.newaxis])
        plt.colorbar(pos0, ax=axs[0,0], location = 'left')
        pos1 = axs[0,1].imshow(sst_data_test_lr[0][...,np.newaxis])
        plt.colorbar(pos1, ax=axs[0,1])
        pos2 = axs[1,0].imshow(ssh_data_test[0][...,np.newaxis])
        plt.colorbar(pos2, ax=axs[1,0], location = 'left')
        pos3 = axs[1,1].imshow(sst_data_test[0][...,np.newaxis])
        plt.colorbar(pos3, ax=axs[1,1])
        plt.show()

    x_test = sst_data_test_lr
    y_test = ssh_data_test_lr

    print('x test shape :', x_test.shape)

    with open('ssh_normalizer.bin', "rb") as f:
        y_normalizer = pickle.load(f)
    with open('sst_normalizer.bin', "rb") as f: 
        x_normalizer = pickle.load(f)

    x_test = x_normalizer.encode(x_test)

    y_test = y_normalizer.encode(y_test)

    x_test = x_test.reshape(ntest,s,s,1)

    print(x_test.shape)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)

    model = torch.load(args.model)
    model.eval()

    if args.superres : 
        x = T.ToTensor()(sst_data_test[0]).permute(1,2,0)[np.newaxis,...]
        y = T.ToTensor()(ssh_data_test[0])

    else : 
        x = sst_data_test_lr[0]
        y = ssh_data_test_lr[0]
        print('x shape :',x.shape)
        x = x[np.newaxis,...,np.newaxis]
        y = y[np.newaxis,...]

    x = x_normalizer.encode(x)
    print(x.shape)
    print('Normalizer : ', y_normalizer.mean, y_normalizer.std)
    
    y, out, test_l2 = predict(x, y, y_normalizer)

    rmsepred = torch.flatten(out[0])
    rmsegt = torch.flatten(y[0])

    print('L2 Loss :', test_l2)
    print("Network prediction : ", math.sqrt(mean_squared_error(rmsepred,rmsegt)))

    
    fig, axs = plt.subplots(2, 2,figsize=(18, 18))
    pos0 = axs[0,0].imshow(ssh_data_test_lr[0][...,np.newaxis])
    plt.colorbar(pos0, ax=axs[0,0], location = 'left')
    pos1 = axs[0,1].imshow(sst_data_test_lr[0][...,np.newaxis])
    plt.colorbar(pos1, ax=axs[0,1])
    pos2 = axs[1,0].imshow(y[0][...,np.newaxis])
    plt.colorbar(pos2, ax=axs[1,0], location = 'left')
    pos3 = axs[1,1].imshow(out[0][...,np.newaxis])
    plt.colorbar(pos3, ax=axs[1,1])
    fig.suptitle(f"RMSE : {math.sqrt(mean_squared_error(rmsepred,rmsegt))}, Training size : {args.size}")
    if args.res :
        plt.show()
    else : 
        plt.savefig(f"/data/jean.legoff/fourier_neural_operator/results_test_i{args.size}.png")
        
    rmses = []
    if args.mrmse : 
        if args.superres : 
            for (ssh,sst) in tqdm(zip(ssh_data_test, sst_data_test)):
            
                x = T.ToTensor()(sst).permute(1,2,0)[np.newaxis,...]
                y = T.ToTensor()(ssh)
            
            
                x = x_normalizer.encode(x)
                y, out, test_l2 = predict(x, y, y_normalizer)
                
                rmsepred = torch.flatten(out[0])
                rmsegt = torch.flatten(y[0])
                
                rmses.append(math.sqrt(mean_squared_error(rmsepred,rmsegt)))
            
        else : 
            
            for (ssh,sst) in tqdm(zip(ssh_data_test_lr, sst_data_test_lr)):
                x = sst
                y = ssh

                x = x[np.newaxis,...,np.newaxis]
                y = y[np.newaxis,...]
                
                x = x_normalizer.encode(x)
                y, out, test_l2 = predict(x, y, y_normalizer)
                
                rmsepred = torch.flatten(out[0])
                rmsegt = torch.flatten(y[0])
                
                rmses.append(math.sqrt(mean_squared_error(rmsepred,rmsegt)))
            
        print('Model MRMSE : ', np.mean(rmses))
        
    if args.plotfour : 
        
        x = T.ToTensor()(sst_data_test[0]).permute(1,2,0)[np.newaxis,...]
        y = T.ToTensor()(ssh_data_test[0])
        
        x = x_normalizer.encode(x)
        
        outs = []
        
        for i in ['16', '48', '144', '432'] : 
            
            model = torch.load('best_fno_sst_'+i+'.pth')
            model.eval()
            
            y, out, test_l2 = predict(x, y, y_normalizer)

            rmsepred = torch.flatten(out[0])
            rmsegt = torch.flatten(y[0])

            print('L2 Loss :', test_l2)
            print("Network prediction : ", math.sqrt(mean_squared_error(rmsepred,rmsegt)))
            
            outs.append(out)

        
        fig, axs = plt.subplots(1, 5,figsize=(18, 36))
        
        axs[0].imshow(outs[0][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[1].imshow(outs[1][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[2].imshow(outs[2][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[3].imshow(outs[3][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[4].imshow(y[0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        '''
        axs[0].imshow(outs[0][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[1].imshow(outs[1][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[2].imshow(outs[2][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[3].imshow(outs[3][0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        axs[4].imshow(y[0][...,np.newaxis],vmin = -0.5,vmax = 0.8)
        '''
        axs[0].title.set_text('Trained at 16')
        axs[1].title.set_text('Trained at 48')
        axs[2].title.set_text('Trained at 144')
        axs[3].title.set_text('Trained at 432')
        axs[4].title.set_text('Ground Truth')
        
        if args.res :
            plt.show()
        else : 
            plt.savefig(f"/data/jean.legoff/fourier_neural_operator/results_test_i{args.size}.png")