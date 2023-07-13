import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import glob
import numpy as np
from random import randint
from i3d import I3D
#from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import os
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers[0]
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, end_point):
        outputs = []
        self.gradients = []
        
    
        if end_point.startswith('Mixed'):
            print('target layer:', self.target_layers)
            b0 = self.model.b0(x)
            if self.target_layers=='b0':
                x.register_hook(self.save_gradient)
                outputs += [x]
                
            b1 = self.model.b1b(self.model.b1a(x))
            if self.target_layers=='b1':
                x.register_hook(self.save_gradient)
                outputs += [x]
                
            b2 = self.model.b2b(self.model.b2a(x))
            if self.target_layers=='b2':
                x.register_hook(self.save_gradient)
                outputs += [x]    
            
            b3 = self.model.b3b(self.model.b3a(x))
            if self.target_layers=='b3':
                x.register_hook(self.save_gradient)
                outputs += [x]
    
            x = torch.cat([b0,b1,b2,b3], dim=1)
            if self.target_layers not in ['b0', 'b1', 'b2', 'b3']:
                x.register_hook(self.save_gradient)
                outputs += [x]
              
        else:
            x = self.model(x)
                            
            x.register_hook(self.save_gradient)
            outputs += [x]
                    
        
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        
        #for name, module in self.model._modules.items():
        for end_point in self.model.VALID_ENDPOINTS:
            if end_point in self.model.end_points:
                print(end_point)  
            #    print(end_point)
                if self.model._modules[end_point] == self.feature_module:
                    print('feature extraction: ', self.feature_module)
                    target_activations, x = self.feature_extractor(x, end_point)
                    
                else:
                    x = self.model._modules[end_point](x)

        if self.feature_module==self.model.avg_pool: 
            print('feature extraction!!!')
            
            #x = self.model.avg_pool(x)
            target_activations, x = self.feature_extractor(x, 'avg_pool')

            feature = self.model.dropout(x)
            x = self.model.conv3d_0c_1x1(feature)
            if self.model._spatial_squeeze:
                x = x.squeeze(3).squeeze(3)

                print('logits', x.shape)
        else:
            #target_activations=[0]
            x = self.model.avg_pool(x)
           
            feature = self.model.dropout(x)

            x = self.model.logits(feature)
            
            x = x.squeeze(3).squeeze(3)
            print('logits', x.shape)
        # logits x is batch X time X classes, which is what we want to work with

        
        return target_activations, x



def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input
    
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_rgb_frames(frames):
    images = []
    for i in frames:
        img=cv2.imread(i)[:, :, [2, 1, 0]]       
        img = cv2.resize(img, (224, 224))
        images.append(img)
    images = np.asarray(images, dtype=np.float32)
    images = (images/255.)*2 - 1
    return images

  
def preprocess_video(img_dir):
    
    frame_indices = []
    images = sorted(glob.glob(img_dir+'/*g'))
    n_frames=len(images)
    sample_duration=64
    step = 2
    
    if (n_frames > sample_duration * step):
        start = randint(0, n_frames - sample_duration*step)
        for i in range(start, start + sample_duration*step, step):
            frame_indices.append(images[i])
    elif n_frames < sample_duration:
        while len(frame_indices) < sample_duration:
            frame_indices.extend(images)
        frame_indices = frame_indices[:sample_duration]
    else:
        start = randint(0, n_frames -sample_duration)
        for i in range(start, start+sample_duration):
            frame_indices.append(images[i])

    
    imgs = load_rgb_frames(frame_indices)
    imgs = video_to_tensor(imgs)
    imgs.unsqueeze_(0)
    imgs = imgs.requires_grad_(True)
    return imgs


def show_cam_on_image(img, mask, i, imdir):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    if not os.path.exists(imdir):
        os.makedirs(imdir) 
    cv2.imwrite(imdir+"/cam_"+str(i).zfill(6)+'.jpg', np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
            
        return self.model(input)
    
    def __call__(self, input, index=None):
        
        if self.cuda:
            features, logits = self.extractor(input.cuda(0))
        else:
            features, logits = self.extractor(input)

        t = input.size(2)

        output = F.upsample(logits, t, mode='linear')
        output = torch.max(output, dim=2)[0]
        print('feature: ', features[0].shape)
        print('output:', output.shape)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3, 4))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :, :]

        cam = np.maximum(cam, 0)
        cams=np.zeros(np.append(cam.shape[0] ,input.shape[3:]))
        for c in range(len(cams)):
            cams[c] = cv2.resize(cam[c], (input.shape[4], input.shape[3]))
            cams[c] = cams[c] - np.min(cams[c])
            cams[c] = cams[c] / np.max(cams[c])
        return cams


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--images-dir', type=str, default='./examples/video1',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)



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
    train_data = mxnet.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset= HMDB51(root='/data/luyi/lsx/datasets/hmdb51/rawframes',setting='/data/luyi/lsx/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt',train=False, new_length=79, transform=transform_train)
    test_data = mxnet.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    # class_list
    classes_list = os.listdir('/data/luyi/lsx/datasets/hmdb51/rawframes')
    classes_list.sort()

    rgb_weights_path='/data/luyi/lsx/i3d/model_weight/model_rgb.pth'#/net/net3.pth'

    # rgb
    print("rgb")
    i3d_rgb = torch.load('/data/luyi/lsx/i3d/net/net3.pth')
    i3d.train(False)
    print(i3d_rgb.mixed_5c._modules.keys())
    

    grad_cam = GradCam(model=i3d_rgb, feature_module=i3d_rgb.mixed_5c, target_layer_names=["branch_0"])

    batch=0
    size=len(test_data)
    print(size)
    # assert 1==2
    loss_sum=0
    correct_sum=0
 
    for x,y in test_data:
        rgb_sample = x.asnumpy().squeeze()
        x=torch.Tensor(rgb_sample)
        print(torch.Tensor(rgb_sample).shape)
        start_time = time.time()

        
        pred, out_logit = i3d_rgb(x.cuda())
    
        mask = grad_cam(x, None)
        imgs=sorted(os.listdir('/data/luyi/lsx/i3d/gradcam'))
        imdir='/data/luyi/lsx/i3d/gradcam'.split('/')[-1]+'_cams'

        imgs = x.permute(1,0,2,3)
        for i, fimg in enumerate(imgs):
            k=i*len(mask)//len(imgs)
            img = dimg

            show_cam_on_image(img, mask[k], i, imdir)
        batch =batch+1
        if batch==1: 
            break