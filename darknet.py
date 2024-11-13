import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from region_loss import RegionLoss
from cfg import *

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        #Simen: edited as suggested here: https://github.com/marvis/pytorch-yolo2/issues/129#issue-350726531
        #x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        #x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        #x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        #x = x.view(B, hs*ws*C, H/hs, W/ws)
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, clp=1.0, dr=1.0, gaussian=False):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]


        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

        #FNS params:
        self.clp, self.dr, self.gaussian=clp,dr,gaussian

    def forward(self, x, p = None, occ='fm', mode='themis', haste=False, fns=False):
        fm_out=False
        ind = -2
        self.loss = None
        outputs = dict()
        #kk=0

        if occ=='input' and p is not None:
            p=p.squeeze(0)
            p = p.cpu().numpy()
            if mode=='spann':
                #print(p)
                #input(p.shape)
                x[:, :, [p[0]], [p[1]]] = 0.0
            else:
                x[:, :, [[i] for i in np.arange(p[0], p[0]+p[2])], [np.arange(p[1], p[1]+p[3])]] = 0.0#torch.zeros((p[2], p[3])).cuda()

        """
        for i, block in enumerate(self.blocks):
            print(block['type'])
            if i >= 1:
                print(self.models[i-1])
            input('hus')
        """
        fm=x
        mps=0
        twfs=0
        fvtw=0
        tentf=0
        for block in self.blocks:
            ind = ind + 1

            #if ind > 0:
            #    return x
            #print(block)
            #input('yes')
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':

                if block['type'] == 'maxpool':
                    mps=mps+1
                    if not fm_out:
                        #print(x.shape)
                        fm = self.models[ind](x)
                        #fm = x
                        #kk=kk+1
                        fm_out=True
                        if haste:
                            return None, fm
                        #print(fm.shape)
                        #print('thi caounta', str(kk))
                        #input('we happy?')
                        #fm_out=kk==2#input('we happy?')=='1'
                        if p is not None and fm_out  and occ=='fm':# and occ=='fm':
                            p=p.squeeze(0)
                            #print(x.shape)
                            p = p.cpu().numpy()
                            #print(p.shape)
                            #input(fm[:, :, [[i] for i in np.arange(p[0], p[0]+p[2])], [np.arange(p[1], p[1]+p[3])]].shape)
                            if mode=='spann':
                                fm[:, :, [p[0]], [p[1]]] = 0.0#torch.zeros((p[2], p[3])).cuda()
                            else:
                                fm[:, :, [[i] for i in np.arange(p[0], p[0]+p[2])], [np.arange(p[1], p[1]+p[3])]] = 0.0#torch.zeros((p[2], p[3])).cuda()
                            #x=fm
                        x = fm#self.models[ind](fm)
                        #print(fm.shape)
                        #print(x.shape)
                        #input('huh?')
                    elif fns and mps>1:#else:
                        x = self.models[ind](self.clamp(x, clp=self.clp, dr=self.dr, gaussian=self.gaussian))
                        #input('your call')
                    else:
                        x = self.models[ind](x)



                elif fns and block['type'] == 'convolutional':
                    if block['activation']=='linear':
                        x = self.models[ind](self.clamp(x, clp=1.1*self.clp, dr=self.dr, gaussian=self.gaussian))
                    elif block['filters']==256:
                        twfs=fvtw+1
                        if twfs==4:
                            x = self.models[ind](self.clamp(x, clp=self.clp, dr=self.dr, gaussian=self.gaussian))
                        else:
                            x = self.models[ind](x)

                    elif block['filters']==512:
                        fvtw=fvtw+1
                        if fvtw==5:
                            x = self.models[ind](self.clamp(x, clp=self.clp, dr=self.dr, gaussian=self.gaussian))
                        else:
                            x = self.models[ind](x)

                    elif block['filters']==1024:
                        fvtw=fvtw+1
                        if fvtw==4:
                            x = self.models[ind](self.clamp(x, clp=self.clp, dr=self.dr, gaussian=self.gaussian))
                        else:
                            x = self.models[ind](x)
                    else:
                        x = self.models[ind](x)
                else:
                    x = self.models[ind](x)
                #print(x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                input("ARISE")
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        #print(x.shape)
        #print(x)
        #print(fm.shape)
        #input('going home?')
        return x, fm

    def print_network(self):
        print_cfg(self.blocks)

    def clamp(self, x, clp=1.0, dr=1.0, gaussian=False):
        norm = torch.norm(x, dim=1, keepdim=True)
        thre = torch.mean(torch.mean(clp * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x / torch.clamp_min(norm, min=1e-7)
        mask = (norm > thre).float()
        if gaussian:
            normd = thre * torch.exp(-1 / thre / thre * (norm - thre) * (norm - thre) * math.log(dr))
        else:
            normd = thre * torch.exp(-1 / thre * (norm - thre) * math.log(dr))
        norm = norm * (1 - mask) + normd * mask
        x = x * norm
        return x

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                #Simen: edit as sugessted here: https://github.com/marvis/pytorch-yolo2/issues/129#issue-350726531
                #pad = (kernel_size-1)/2 if is_pad else 0
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
