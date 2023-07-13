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
batchsize=16
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
            print(f"loss: {loss1:>7f} acc: {(correct_sum/(batch*batchsize)):>7f} [{current:>5d}/{size:>5d}]")
        batch=batch+1
    loss_train.append(loss_sum/batch)
    correct_train.append(correct_sum/(batch*batchsize))


def test_loop(data_loader, model, loss_fn):
    batch=0
    size=len(data_loader)
    loss_sum=0
    correct_sum=0
    with torch.no_grad():
        for x,y in data_loader:
            rgb_sample = x.asnumpy().squeeze()
            out_var, out_logit = model(torch.Tensor(rgb_sample).cuda())
            out_tensor = out_var.data.cpu()

            top_val, top_idx = torch.sort(out_tensor, 1, descending=True)
            out_label=top_idx[:,0]
            loss = loss_fn(out_logit, torch.tensor(y.asnumpy()).cuda())
       
            loss_sum+=loss.cpu().sum()
            correct_sum+=(y.asnumpy()==out_label.numpy()).sum()
            batch=batch+1
            
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

    rgb_weights_path='/data/luyi/lsx/i3d/model_weight/model_rgb.pth'

    # rgb
    print("rgb")
    i3d_rgb = I3D(num_classes=len(classes_list), modality='rgb')#len(classes_list)
    # weights initialize
    orginal_dict = i3d_rgb.state_dict() #当前网络的权重字典。
    weight_dict = torch.load(rgb_weights_path) #读取的网络权重字典
    # 通过形状相同，把orignal_dict对应的tensor 换成 weight_dict的tensor。
    i=0

    list1=list(orginal_dict.keys())
    list2=list(weight_dict.keys())
    print(len(list1))
    print(len(list2))
    list1_not_in_list2 = [i for i in list1 if i not in list2]

    list2_not_in_lis1 = [i for i in list2 if i not in list1]
    print(len(list1_not_in_list2))
    print("**************************************************")
    print(len(list2_not_in_lis1))
    for key2,value2 in weight_dict.items():
        if value2.size() == orginal_dict[key2].size():
            i=i+1
            orginal_dict[key2] = weight_dict[key2] # 将orginal换成weight_dict
    i3d_rgb.load_state_dict(orginal_dict)
    print(i)
    # 迁移学习，冻结
    for name, param in i3d_rgb.named_parameters():
        if "conv3d_0c_1x1" in name:
            print(type(param))
            print(name)
            param.requires_grad = True
            print("*****************true")
        else:
            param.requires_grad = False

    i3d_rgb.train()
    i3d_rgb.cuda()

    epochs=20
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(i3d_rgb.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, eps=1e-5)

    
    # example
    # for x,y in train_data:
    #     break
    # rgb_sample = x.asnumpy().squeeze()
    # print(rgb_sample.shape)
    # out_rgb_logit = get_scores(torch.Tensor(rgb_sample), i3d_rgb)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop( train_data,  i3d_rgb, loss_fn, optimizer)
        print("test")
        accuracy=test_loop(train_data,  i3d_rgb, loss_fn)
        if t==10:
            learning_rate=learning_rate*0.1
            optimizer = torch.optim.SGD(i3d_rgb.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
   
        scheduler.step(accuracy)

    torch.save(i3d_rgb, 'net.pth')  # 保存整个神经网络的模型结构以及参数
    torch.save(i3d_rgb.state_dict(), 'net_params.pkl')  # 同上

    plt.title('loss-epoches')
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.plot(loss_train,color="black",label="loss_train")
    plt.plot(loss_test,color="red",label="loss_test")
    plt.legend()
    plt.show()
    
    plt.xlabel("epoches")
    plt.ylabel("accuracy")
    plt.plot(correct_train,color="black",label="loss_train")
    plt.plot(correct_test,color="red",label="loss_test")
    plt.legend()
    plt.show()