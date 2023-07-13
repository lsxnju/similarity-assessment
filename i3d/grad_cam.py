import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import cv2


class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers[0]
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
         
        b0 = self.model.branch_0(x)
        if self.target_layers=="branch_0":
                x.register_hook(self.save_gradient)
                outputs += [x]

        b1 = self.model.branch_1(x)
        if self.target_layers=="branch_1":
                x.register_hook(self.save_gradient)
                outputs += [x]

        b2 = self.model.branch_2(x)
        if self.target_layers=="branch_2":
                x.register_hook(self.save_gradient)
                outputs += [x]

        b3 = self.model.branch_3(x)
        if self.target_layers=="branch_3":
                x.register_hook(self.save_gradient)
                outputs += [x]

        x = torch.cat([b0,b1,b2,b3], dim=1)
        return outputs, x

class ModelOutputs():

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                print('feature extraction: ', self.feature_module)
                target_activations, x = self.feature_extractor(x)
            else:
                x = module(x)

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
            # x = self.model.avg_pool(x)
           
            # feature = self.model.dropout(x)

            # x = self.model.logits(feature)
            
            x = x.squeeze(3).squeeze(3)
            print('logits', x.shape)
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.feature_module = feature_module
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input.cuda())

        t=input.size(2)

        output = F.upsample(output, t, mode='linear')
        output = torch.max(output, dim=2)[0]
        print('feature: ', features[0].shape)
        print('output:', output.shape)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

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
        # B, C, Tg, _, _ = grads_val.size()
        # weights = torch.mean(grads_val.view(B, C, Tg, -1), dim=3)
        # weights = weights.view(B, C, Tg, 1, 1)
        # activations = features[-1]
        # localization_map = torch.sum(
        #     weights * activations, dim=1, keepdim=True)
        # localization_map = F.relu(localization_map)
        # localization_map = F.interpolate(
        #     localization_map,
        #     size=(79,224, 224),
        #     mode="trilinear",
        #     align_corners=False,
        # )
        # localization_map_min, localization_map_max = (
        #     torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
        #            0
        #     ],
        #     torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
        #         0
        #     ],
        # )
        # localization_map_min = torch.reshape(
        #     localization_map_min, shape=(B, 1, 1, 1, 1)
        # )
        # localization_map_max = torch.reshape(
        #     localization_map_max, shape=(B, 1, 1, 1, 1)
        # )
        #     # Normalize the localization map.
        # localization_map = (localization_map - localization_map_min) / (
        #     localization_map_max - localization_map_min + 1e-6
        # )
        # localization_map = localization_map.data
        # return localization_map


