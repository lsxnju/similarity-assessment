import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from i3d import I3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def get_scores(sample, model):
    out_var, out_logit = model(sample)
    out_tensor = out_var.data.cpu()

    top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

    print(
        'Top {} classes and associated probabilities: '.format(5))
    for i in range(5):
        if top_idx[0, i]>51:
            continue
        print('[{}]: {:.6E}'.format(classes_list[top_idx[0, i]],
                                    top_val[0, i]))
    return out_logit

def train_loop(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, data in enumerate(data_loader, 0):
        x, y = data
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batches, data in enumerate(data_loader, 0):
            x, y = data
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


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
    train_data = mxnet.gluon.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
    test_dataset= HMDB51(root='/data/luyi/lsx/datasets/hmdb51/rawframes',setting='/data/luyi/lsx/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt',train=False, new_length=79, transform=transform_train)
    test_data = mxnet.gluon.data.DataLoader(test_dataset, batch_size=25, shuffle=True)
    classes_list = os.listdir('/data/luyi/lsx/datasets/hmdb51/rawframes')
    classes_list.sort()

    epochs = 100
    rgb_weights_path='/data/luyi/lsx/i3d/model_weight/model_rgb.pth'
    flow_weights_path='/data/luyi/lsx/i3d/model_weight/model_flow.pth'
    # rgb
    print("rgb")
    i3d_rgb = I3D(num_classes=len(classes_list), modality='rgb')
    i3d_rgb.eval()
    # i3d_rgb.load_state_dict(torch.load(rgb_weights_path))
    

    orginal_dict = i3d_rgb.state_dict() #当前网络的权重字典。
    weight_dict = torch.load(rgb_weights_path) #读取的网络权重字典
    # 通过形状相同，把orignal_dict对应的tensor 换成 weight_dict的tensor。
    for key,value in orginal_dict.items():
        for key2,value2 in weight_dict.items():
            if value2.size() == value.size():
                orginal_dict[key] = weight_dict[key2] # 将orginal换成weight_dict
    i3d_rgb.load_state_dict(orginal_dict)
    for name, param in i3d_rgb.named_parameters():
        if "conv3d_0c_1x1" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for x,y in train_data:
        break
    rgb_sample = x.asnumpy().squeeze()
    print(rgb_sample.shape)
    out_rgb_logit = get_scores(torch.Tensor(rgb_sample), i3d_rgb)

    # # flow
    # print("flow")
    # i3d_flow = I3D(num_classes=len(classes_list), modality='flow')
    # i3d_flow.eval()
    # i3d_flow.load_state_dict(torch.load(flow_weights_path))
    # # i3d_flow.cuda()

    # flow_sample = np.load(args.flow_sample_path).transpose(0, 4, 1, 2, 3)
    # print(flow_sample.shape)
    # out_flow_logit = get_scores(flow_sample, i3d_flow)
    # model = I3D()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, eps=1e-5)

    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train_loop(train, model, loss_fn, optimizer)
    #     accuracy = test_loop(test, model, loss_fn)
    #     scheduler.step(accuracy)
'''
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
torch.save(model, 'model.pth')'''
