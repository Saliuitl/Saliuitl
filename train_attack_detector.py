##############################################################################
# Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
##############################################################################

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import nets.attack_detector
import PIL

#from scripts.progress_bar import progress_bar
from helper import clustering_data_preprocessing
import numpy as np
import joblib

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import copy
import random
from scipy.spatial import distance_matrix
parser = argparse.ArgumentParser(description='PyTorch AtkDet Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--optimizer', default='adam', type=str, help='optimization algorithm')
parser.add_argument('--dataset', default='cifar', type=str, help='dataset to train/test on')
parser.add_argument('--remove', default='_', type=str, help='dataset to train/test on')
parser.add_argument('--base_dataset', default=None, type=str, help='dataset the model was trained on')
parser.add_argument('--ensemble_step', default=5, type=int, help='100/threshold set size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', action='store_true', help='do not train!')
parser.add_argument('--train_aug', action='store_true', help='randomly translate curves during training')
parser.add_argument('--pre_process', action='store_true', help='translate curves before feeding them into the model')
parser.add_argument('--type_pre', default='nclusters', type=str, help='feature for pre-processing curves')

parser.add_argument("--feature_maps",default="fms.npy",type=str,help="filename with fms")
parser.add_argument("--adv_feature_maps",default="pfms.npy",type=str,help="filename with pfms")
parser.add_argument("--tags",default="1p_train_tags.npy",type=str,help="all anmes")
parser.add_argument("--cleanarr",default="clean_gap.npy",type=str,help="clean names")

parser.add_argument('--train_frac', default=0.8, type=float, help='fraction of dataset to be used for training')
parser.add_argument("--clip",default=-1,type=int)
parser.add_argument("--npatch_base",default=-1,type=int)

parser.add_argument("--savedir",default='NN_based_objdet/',type=str,help="save path")

parser.add_argument("--cfg",default="cfg/yolo.cfg",type=str,help="relative directory to cfg file")
parser.add_argument("--weightfile",default="weights/yolo.weights",type=str,help="path to checkpoints")

parser.add_argument("--model",default='2dcnn_raw',type=str,help="architecture for detection")

parser.add_argument("--dbscan_eps", default=1.0, type=float, help="how close to cluster two neurons?")
parser.add_argument("--dbscan_min_pts", default=4, type=int, help="how many neurons is a cluster?")
parser.add_argument("--bin_mag", default=1.0, type=float, help="magnitude of binarized neurons")
parser.add_argument("--rf", default=9, type=float, help="receptive field proportion")


parser.add_argument("--scale_var",action='store_true',help="standardize variance before clustering")
parser.add_argument("--scale_mean",action='store_true',help="center mean before clustering")

args = parser.parse_args()
mn, std= args.scale_mean, args.scale_var

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

fms=np.load(args.feature_maps)
pfms=np.load(args.adv_feature_maps)

"""
Get ensemble attribute vectors and labels (0=clean, 1=attack)
"""
print("Computing ensemble attribute vectors...")
all_x, all_y=[], []
for group, label in zip([fms, pfms], [0.0, 1.0]):
    for k, fm in enumerate(group):
        cluster_curve, distance_curve, distance_sd_curve=[], [], []
        imp_neu_curve = []

        for beta in [0.0+x*0.01 for x in range(0,100,args.ensemble_step)]:#cifar until beta=0.45, imgnet until beta=0.25
            binarized_fm=np.array(fm>=np.max(fm)*beta, dtype='float32')
            imp_neu_curve.append(binarized_fm.sum()*(not 'impneu' in args.remove))
            if 'nclus' in args.remove and 'avg' in args.remove and 'std' in args.remove:
                continue
            x,y=np.where(binarized_fm>0)
            thing=np.hstack((x.reshape(-1,1),y.reshape(-1,1)))#,binarized_fm.flatten().reshape(-1,1)))
            thing = StandardScaler(with_mean=mn, with_std=std).fit_transform(thing)
            #datas.append(thing)
            cluster=DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_pts).fit(thing)
            data=thing
            clusters= np.unique(cluster.labels_)#, return_counts=True)
            nclusters=len([x!=-1 for x in clusters])*(not 'nclus' in args.remove)
            cluster_curve.append(nclusters)
            #continue
            centroids=[]
            mx_ic_d,mn_ic_d,avg_ic_d=[],[],[]
            for cluster_label in clusters:
                if cluster_label==-1:
                    continue
                data_c=data[np.where(cluster.labels_==cluster_label)]
                data_samp=data_c[np.random.choice([i for i in range(len(data_c))], size=min(1000, len(data_c)), replace=False)]
                dmx=distance_matrix(data_samp, data_samp)
                dmx=dmx[np.tril_indices(dmx.shape[0], k=-1)]
                if len(dmx):
                    avg_ic_d.append(np.mean(dmx))
            if len(avg_ic_d):
                avg_intracluster_d=np.mean(avg_ic_d)*(not 'avg' in args.remove)
                avg_intracluster_std=np.std(avg_ic_d)*(not 'std' in args.remove)
            else:
                avg_intracluster_d=0
                avg_intracluster_std=0
            distance_curve.append(avg_intracluster_d)
            distance_sd_curve.append(avg_intracluster_std)

        feat_stack=[]

        if not 'nclus' in args.remove:
            feat_stack.append(np.array(cluster_curve).reshape(-1,1))
        if not 'avg' in args.remove:
            feat_stack.append(np.array(distance_curve).reshape(-1,1))
        if not 'std' in args.remove:
            feat_stack.append(np.array(distance_sd_curve).reshape(-1,1))
        if not 'impneu' in args.remove:
            feat_stack.append(np.array(imp_neu_curve).reshape(-1,1))
        curve_data=np.expand_dims(np.hstack(feat_stack), axis=0)#expand_dims(axis=0)
        all_x.append(curve_data)
        all_y.append(label)
"""
Structure into training/validation splits
"""
train_ind=np.random.choice([i for i in range(len(all_x))], size=int(args.train_frac*len(all_x)), replace=False)
val_ind=[i for i in range(len(all_x)) if i not in train_ind]
train_x=np.vstack(all_x)[train_ind]
val_x=np.vstack(all_x)[val_ind]
train_y=np.vstack(all_y)[train_ind]
val_y=np.vstack(all_y)[val_ind]
train_xx, train_yy = 2*nn.functional.normalize(torch.Tensor(clustering_data_preprocessing(train_x, skip=True)), dim=2, p=float('inf')) - 1, torch.Tensor(train_y)
val_xx, val_yy = 2*nn.functional.normalize(torch.Tensor(clustering_data_preprocessing(val_x, skip=True)), dim=2, p=float('inf')) - 1, torch.Tensor(val_y)


trainset = torch.utils.data.TensorDataset(train_xx, train_yy) # create your datset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
valset = torch.utils.data.TensorDataset(val_xx, val_yy)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

"""
Build and train AD
"""
# Model
print('==> Building model..')
if args.remove is None:
    remstr=''
else:
    remstr=args.remove
if args.base_dataset is None:
    pth_path = './checkpoints/' + args.model +'_' + args.dataset + '_' + str(args.ensemble_step) + '_' + remstr + '_atk_det.pth'
else:
    pth_path = './checkpoints/' + args.model +'_' + args.base_dataset + '_' + str(args.ensemble_step) + '_' + remstr +'_atk_det.pth'

if args.model=='1dcnn':
    net = nets.attack_detector.attack_detector()
elif args.model=='2dcnn':
    net = nets.attack_detector.cnn()

#main model for detection, note despite the misleading name it is a 1D-CNN (in_feats = number of ensemble attributes n_f )
elif args.model=='2dcnn_raw':
    net = nets.attack_detector.cnn_raw(in_feats=(not 'impneu' in args.remove) + (not 'nclus' in args.remove) +
                                                (not 'avg' in args.remove) + (not 'std' in args.remove))
elif args.model=='mlp':
    net = nets.attack_detector.mlp()
elif args.model=='mlp+':
    net = nets.attack_detector.mlp(in_size=4)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(pth_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
if args.optimizer=='sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.optimizer=='adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted=torch.round(outputs)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Loss: {} | Acc: {} ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, best_acc=0, best_loss=np.inf, val=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx_list=[]
    if val:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                predicted=torch.round(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print('Loss: {} | Acc: {} ({}/{})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
        if not args.test and test_loss<=best_loss:#acc=>best_acc:#acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, pth_path)
    else:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                predicted=torch.round(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Loss: {} | Acc: {} ({}/{})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
        if not args.test and acc>best_acc:#acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, pth_path)

    return acc, test_loss

best_acc=0
best_loss=np.inf
no_improve=0
for epoch in range(start_epoch, start_epoch+1500):
    if not args.test:
        train(epoch)
        current_acc, current_loss=test(epoch, best_acc, best_loss)
        if best_loss>=current_loss:#best_acc < current_acc:
            #best_acc=current_acc
            best_loss=current_loss
            no_improve=0
        else:
            no_improve = no_improve + 1
    if args.test or no_improve > 200:
        #test(epoch, val=False)
        break
