import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from i3d import I3D
import numpy as np
from matplotlib import pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
loss_train=[]
correct_train=[]
loss_test=[]
correct_test=[]
batchsize=1
def get_scores(sample, model):
    out_var, out_logit = model(sample)
    out_tensor = out_var.data.cpu()

    top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

    # print(
    #     'Top {} classes and associated probabilities: '.format(5))
    # for i in range(5):
    #     if top_idx[0, i]>51:
    #         continue
    #     print('[{}]: {:.6E}'.format(classes_list[top_idx[0, i]],
    #                                 top_val[0, i]))
    print(top_idx[:,0])
    return out_logit,top_idx[:,0]

def train_loop(data_loader, model, loss_fn, optimizer):
    batch=0
    size=len(data_loader)*batchsize
    loss_sum=0
    correct_sum=0
    if(len(data_loader)<batchsize):
        return
    for x,y in data_loader:
        rgb_sample = x.asnumpy().squeeze()
        out_var, out_logit = i3d_rgb(torch.Tensor(rgb_sample).cuda())
        out_tensor = out_var.data.cpu()

        top_val, top_idx = torch.sort(out_tensor, 1, descending=True)
        out_label=top_idx[:,0]
        loss = loss_fn(out_logit, torch.tensor(y.asnumpy()).cuda())
       

        # Backpropagation
        optimizer.zero_grad()
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.cpu().sum()
        correct_sum+=(y.asnumpy()==out_label.numpy()).sum()

        if batch % 10 == 0:
            loss1, current = loss.item(), batch * batchsize
            # print(f"loss: {loss1:>7f} [{current:>5d}/{size:>5d}]")
        batch=batch+1
    loss_train.append(loss_sum/batch)
    correct_train.append(correct_sum/(batch*batchsize))


def test_loop(data_loader, model, loss_fn):
    batch=0
    size=len(data_loader)
    print(size)
    # assert 1==2
    loss_sum=0
    correct_sum=0
    with torch.no_grad():
        for x,y in data_loader:
            rgb_sample = x.asnumpy().squeeze(0)
            print(torch.Tensor(rgb_sample).shape)

            out_var, out_logit = model(torch.Tensor(rgb_sample).cuda())
            out_tensor = out_var.data.cpu()

            top_val, top_idx = torch.sort(out_tensor, 1, descending=True)
            out_label=top_idx[:,0]
            loss = loss_fn(out_logit, torch.tensor(y.asnumpy()).cuda())
       
            loss_sum+=loss.cpu().sum()
            correct_sum+=(y.asnumpy()==out_label.numpy()).sum()
            print(correct_sum)
            batch=batch+1
            if batch==1: 
                break
    loss_test.append(loss_sum/batch)
    correct_test.append(correct_sum/(batch*batchsize))
    print ( f"Test Error: \n Accuracy: {(100 * correct_sum/(batch*batchsize)):>0.1f}%, Avg loss: {(loss_sum/batch):>8f} \n")
    return correct_sum/size


if __name__ == '__main__':
    import os
    # datasets
    from gluoncv.data import HMDB51
    import mxnet.gluon.data
    from mxnet.gluon.data.vision import transforms
    from gluoncv.data.transforms import video

    transform_train = transforms.Compose([
        video.VideoCenterCrop(size=224),
        video.VideoToTensor()
    ])
    
    train_dataset = HMDB51(root='/data/luyi/lsx/datasets/hmdb51/rawframes',setting='/data/luyi/lsx/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_train_split_1_rawframes.txt',train=True, new_length=79, transform=transform_train)
    train_data = mxnet.gluon.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataset= HMDB51(root='/data/luyi/lsx/datasets/hmdb51/rawframes',setting='/data/luyi/lsx/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt',train=False, new_length=79, transform=transform_train)
    test_data = mxnet.gluon.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    # class_list
    classes_list = os.listdir('/data/luyi/lsx/datasets/hmdb51/rawframes')
    classes_list.sort()

    rgb_weights_path='/data/luyi/lsx/i3d/model_weight/model_rgb.pth'#/net/net3.pth'

    # rgb
    print("rgb")
    i3d_rgb = torch.load('/data/luyi/lsx/i3d/net/net3.pth')

    # 迁移学习，冻结
    for name, param in i3d_rgb.named_parameters():
            param.requires_grad = False

    i3d_rgb.cuda()

    epochs=20
    loss_fn = nn.CrossEntropyLoss()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        accuracy=test_loop(train_data,  i3d_rgb, loss_fn)
        print(accuracy)


    plt.title('loss-epoches')
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.plot(loss_test,color="red",label="loss_test")
    plt.legend()
    plt.show()
    
    plt.xlabel("epoches")
    plt.ylabel("accuracy")
    plt.plot(correct_test,color="red",label="loss_test")
    plt.legend()
    plt.show()





