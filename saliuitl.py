import sys
import time
import os
import math
import copy
import warnings
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils import model_zoo
from torchvision import datasets, transforms
import PIL
from PIL import Image, ImageDraw

import nets.resnet
import nets.attack_detector
from darknet import *
from helper import *

import json
from tqdm import tqdm
import argparse
import joblib

from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from skimage.restoration import inpaint

parser = argparse.ArgumentParser()
parser.add_argument("--save",action='store_true',help="save results to txt")
parser.add_argument("--savedir",default='NN_based_imgclass/',type=str,help="save path")
#inpainting mode:
parser.add_argument("--inpaint",default="biharmonic",type=str, choices=('zero','mean', 'biharmonic', 'diffusion', 'oracle'), help="inpainting method")
#victim model (obj. det.)
parser.add_argument("--cfg",default="cfg/yolo.cfg",type=str,help="relative directory to cfg file")
parser.add_argument("--weightfile",default="weights/yolo.weights",type=str,help="path to YOLOv2 checkpoints")

parser.add_argument("--performance",action='store_true',help="save recovery performance (time) per frame")
parser.add_argument("--performance_det",action='store_true',help="save detection performance (time) per frame")

parser.add_argument("--effective_files",default=None,type=str,help="file with list of effective adv examples")
parser.add_argument("--geteff",action='store_true',help="save array with effective attack names")
parser.add_argument("--uneffective",action='store_true',help="use only non-effective attacks")


parser.add_argument("--clean",action='store_true',help="use only clean images")
parser.add_argument("--bypass_det",action='store_true',help="skip detection stage")
parser.add_argument("--bypass",action='store_true',help="skip recovery stage")

parser.add_argument("--lim",default=1000000,type=int,help="limit on number of images/frames to process")
parser.add_argument('--imgdir', default="inria/Train/pos", type=str,help="path to clean data")
parser.add_argument('--patch_imgdir', default="inria/Train/pos", type=str,help="path to adversarially patched version of data")
parser.add_argument('--ground_truth', default=None, type=str,help="path to ground truth labels (applies only to image classification datasets)")
parser.add_argument("--neulim",default=0.5,type=float,help="what fraction of input pixels are the max. that should be occluded?")

parser.add_argument('--dataset', default='inria', choices=('inria','voc','imagenet','cifar'),type=str,help="dataset")
parser.add_argument("--det_net",default='2dcnn_raw',type=str,help="architecture of attack detector AD")
parser.add_argument("--det_net_path",default='checkpoints/2dcnn_raw_imagenet_atk_det.pth',type=str,help="path to trained AD weights")
parser.add_argument("--nn_det_threshold",default=0.5,type=float,help="decision threshold for NN detector (alpha* in paper)")
parser.add_argument("--iou_thresh",default=0.5,type=float,help="iou threshold for effective attack definition/evaluation")

parser.add_argument("--save_scores",action='store_true',help="save detection scores AD(s) - useful to evaluate attack detection")
parser.add_argument("--save_outcomes",action='store_true',help="save recovery outcomes (for obj. detection these are bounding boxes, for img. classification a binary label indicating whether the output matches the ground truth)")
parser.add_argument("--n_patches",default='1',type=str,help="number of patches (just an informative string to save results)")

parser.add_argument("--dbscan_eps", default=1.0, type=float, help="how close to cluster two neurons?  - default is 1 (only adjacent neurons)")
parser.add_argument("--dbscan_min_pts", default=4, type=int, help="how many neurons is a cluster?")

parser.add_argument("--scale_var",action='store_true',help="standardize variance before clustering")
parser.add_argument("--scale_mean",action='store_true',help="center mean before clustering")

parser.add_argument('--ensemble_step', default=5, type=int, help='100/threshold set size')
parser.add_argument('--inpainting_step', default=5, type=int, help='100/threshold set size')
parser.add_argument("--eval_class",action='store_true',help="match class for iou evaluation")

parser.add_argument('--remove', default='_', type=str, help='optionally omit clustering features indicated as feat1_feat2_feat3 (ablation study)')
args = parser.parse_args()

warnings.filterwarnings("ignore")
print("Setup...")
if args.dataset in ['cifar', 'imagenet']:
    model = nets.resnet.resnet50(pretrained=True,clip_range=None,aggregation=None)
elif args.dataset in ['inria', 'voc']:
    cfgfile = args.cfg
    weightfile = args.weightfile
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model = model.eval().cuda()
    img_size = model.height
    ious={'clean':[], 'random':[], 'adversarial':[]}

mn, std= args.scale_mean, args.scale_var
imgdir=args.imgdir
patchdir=args.patch_imgdir
device = 'cuda'
#build and initialize model
if args.dataset == 'imagenet':
    ds_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    denorm=transforms.Normalize([-0.485/.229, -0.456/.224, -0.406/.225], [1/0.229, 1/0.224, 1/0.225])
    ds_transforms_patch = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ds_transforms_inp = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
elif args.dataset == 'cifar':
    ds_transforms = transforms.Compose([
        transforms.Resize(192),#
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    denorm=transforms.Normalize([-0.4914/.2023, -0.4822/.1994, -0.4465/.2010], [1/0.2023, 1/0.1994, 1/0.2010])
    ds_transforms_patch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds_transforms_inp = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('./checkpoints/resnet50_192_cifar.pth')
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()

#if torch.cuda.is_available() else 'cpu'

if args.det_net=='1dcnn':
    net = nets.attack_detector.attack_detector()#net = nets.resnet.resnet50(pretrained=True)
elif args.det_net=='2dcnn':
    net = nets.attack_detector.cnn()#net = nets.resnet.resnet50(pretrained=True)
elif args.det_net=='2dcnn_raw':
    net = nets.attack_detector.cnn_raw(in_feats=(not 'impneu' in args.remove) + (not 'nclus' in args.remove) +
                                                (not 'avg' in args.remove) + (not 'std' in args.remove))
elif args.det_net=='mlp':
    net = nets.attack_detector.mlp()#net = nets.resnet.resnet50(pretrained=True)
elif args.det_net=='mlp+':
    net = nets.attack_detector.mlp(in_size=4)#net = nets.resnet.resnet50(pretrained=True)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
print('loading detector net')
assert os.path.isdir('./checkpoints'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.det_net_path)
net.load_state_dict(checkpoint['net'])
net.eval()



all_atk=[]
clean_corr=0
detected=0
success_atk=0
kount=0
iou_thresh=args.iou_thresh
mask_ious=[]

if args.effective_files!=None:
    eff_files=list(np.load(os.path.join(patchdir, args.effective_files)))
    eff_files=[x.split('.')[0] for x in eff_files]
else:
    eff_files = None

if args.save_scores:
    score_array=[]
if args.save_outcomes:
    outcome_array=[]
if args.performance_det:
    perf_array_clus=[]
    perf_array_det=[]
if args.performance:
    perf_array=[]
    perf_array_cluz=[]

def beta_iteration(beta, fm, raw=True):

    binarized_fm=np.array(fm>=np.max(fm)*beta, dtype='float32')

    x,y=np.where(binarized_fm>0)
    thing=np.hstack((x.reshape(-1,1),y.reshape(-1,1)))#,binarized_fm.flatten().reshape(-1,1)))
    thing = StandardScaler(with_mean=mn, with_std=std).fit_transform(thing)
    cluster=DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_pts).fit(thing)

    #data1=thing
    clusters= np.unique(cluster.labels_)#, return_counts=True)

    #continue
    #centroids=[]
    avg_ic_d=[]
    biggie=[]
    for cluster_label in clusters:
        if cluster_label==-1:
            continue
        data_c=thing[np.where(cluster.labels_==cluster_label)]
        biggie.append(data_c)
        data_samp=data_c[np.random.choice([i for i in range(len(data_c))], size=min(1000, len(data_c)), replace=False)]
        dmx=distance_matrix(data_samp, data_samp)
        dmx=dmx[np.tril_indices(dmx.shape[0], k=-1)]
        if len(dmx):
            avg_ic_d.append(np.mean(dmx))
    if len(avg_ic_d):
        avg_intracluster_d=np.mean(avg_ic_d)
        avg_intracluster_std=np.std(avg_ic_d)
    else:
        avg_intracluster_d=0
        avg_intracluster_std=0

    if not raw:
        feat_stack=[]

        if not 'nclus' in args.remove:
            feat_stack.append(len([x!=-1 for x in clusters]))
        if not 'avg' in args.remove:
            feat_stack.append(avg_intracluster_d)
        if not 'std' in args.remove:
            feat_stack.append(avg_intracluster_std)
        if not 'impneu' in args.remove:
            feat_stack.append(binarized_fm.sum())
        #return biggie, [len([x!=-1 for x in clusters]), avg_intracluster_d, avg_intracluster_std, binarized_fm.sum()]
        return biggie, feat_stack
    return biggie

val_dataset=sorted(os.listdir(imgdir)[:min(args.lim, len(os.listdir(imgdir)))])

if args.dataset in ['inria', 'voc']:
    for imgfile in tqdm(val_dataset):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            nameee=imgfile.split('.')[0]
            if (eff_files != None and nameee not in eff_files and not args.uneffective) or (eff_files != None and args.uneffective and nameee in eff_files):
                continue
            patchfile = os.path.abspath(os.path.join(patchdir, imgfile))
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            padded_img = Image.open(imgfile).convert('RGB')
            w,h=padded_img.size
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            #in case oracle inpainting is tested
            if args.inpaint=='oracle':
                cheat=img_fake_batch.detach().clone()
            clean_boxes, feature_map = do_detect(model, img_fake_batch, 0.4, 0.4, True, direct_cuda_img=True)
            clean_boxes=clean_boxes
            #if nothing is detected in the clean version of the image...
            if not(len(clean_boxes)):
                continue
            kount=kount+1

            cbb=[]
            for cb in clean_boxes:
                cbb.append([T.detach() for T in cb])

            if not args.clean:
                data = Image.open(patchfile).convert('RGB')
                patched_img_cpu=data
                patched_img = transform(patched_img_cpu).cuda()
                p_img = patched_img.unsqueeze(0)
                adv_boxes, feature_map  = do_detect(model, p_img, 0.4, 0.4, True, direct_cuda_img=True)
                adb = []
                for ab in adv_boxes:
                    adb.append([T.detach() for T in ab])
            else:
                p_img=img_fake_batch
                candigatos=[]
                adv_boxes=clean_boxes
                adb=cbb

            """
            DETECTION STAGE
            """
            if args.bypass_det:
                condition=True
            else:# and not analysed:
                if args.performance_det:
                    start=time.process_time()
                fm_np=feature_map[0].detach().cpu().numpy()
                fm=np.sum(fm_np,axis=0)
                biginfo=[]
                vector_s=[]
                for beta in [0.0+x*0.01 for x in range(0,100,args.ensemble_step)]:#cifar until beta=0.45, imgnet until beta=0.25
                    biggie, clus_feats=beta_iteration(beta, fm, raw=False)
                    vector_s.append(clus_feats)
                    biginfo.append(biggie)
                if args.performance_det:
                    end=time.process_time()
                    perf_array_clus.append(end-start)
                vector_s=np.array(vector_s).reshape((1, len(vector_s), len(clus_feats)))
                detector_input = 2*nn.functional.normalize(torch.Tensor(clustering_data_preprocessing(vector_s, skip=True)), dim=2, p=float('inf')) - 1

                with torch.no_grad():
                    detector_output=net(detector_input.to(device))
                detection_score=detector_output.detach().cpu().numpy()
                condition=detection_score>=args.nn_det_threshold
                if args.performance_det:
                    end=time.process_time()
                    perf_array_det.append(end-start)
                if args.save_scores:
                    score_array.append(detection_score)
                if condition and args.bypass:
                    detected=detected+1
            """
            RECOVERY STAGE
            """
            if condition and not args.bypass:
                if args.performance:
                    klust=0
                    start=time.process_time()
                my_mask=np.zeros((416,416))
                detected=detected+1
                stop=False
                if not args.bypass_det:
                    revran_det=[0.0+x*0.01 for x in range(0,100,args.ensemble_step)]
                else:
                    if args.performance:
                        sk=time.process_time()
                    fm_np=feature_map[0].detach().cpu().numpy()
                    fm=np.sum(fm_np,axis=0)
                    revran_det=[]
                    if args.performance:
                        klust=klust + time.process_time()-sk
                revran=list(reversed([0.0+x*0.01 for x in range(0,100,args.inpainting_step)]))[:-1]
                corrected=False
                fm_og=copy.deepcopy(fm)
                vikt=False

                sd_boxes=[]
                bfm_old=np.ones(fm.shape)
                for numel, beta_big in enumerate(revran):
                    #print("nore")
                    if beta_big in revran_det:
                        idx=revran_det.index(beta_big)
                        this=biginfo[idx]
                    else:
                        if args.performance:
                            sk=time.process_time()
                        this=beta_iteration(beta_big, fm_og)
                        if args.performance:
                            klust=klust+time.process_time()-sk

                    if not(len(this)):
                        continue

                    dirso=[]
                    masking=0
                    for d in this:
                        #clusters should have 4+ elements
                        if len(d)<4:
                            continue
                        dirso.append(d)
                        masking=masking+len(d)

                    if not len(dirso):#max(mref)<4:
                        continue
                    bfm=fm_og>=np.max(fm_og)*beta_big

                    #stopping condition number 1:
                    if masking > (args.neulim*my_mask.size)/9:#covering more than half the image is too much
                        continue
                    imgneer=np.zeros((416,416))

                    for mi in range(len(dirso)):
                        for x,y in dirso[mi]:
                            x,y = int(x), int(y)
                            imgneer[max(2*x-1,0):min(2*x+1,416)+1, max(2*y-1,0):min(2*y+1,416)+1]=bfm_old[x,y]

                    bfm_old=bfm
                    p=np.where(imgneer>0.0)
                    in_img=p_img.detach().clone()
                    #Saliuitl*
                    if args.inpaint=='oracle':
                        in_img[:, :, [p[0]], [p[1]]] = cheat[:, :, [p[0]], [p[1]]]#0.0
                    #Saliuitl-Z
                    elif args.inpaint=='zero':
                        in_img[:, :, [p[0]], [p[1]]] = 0.0
                    #Saliuitl
                    elif args.inpaint=='biharmonic':
                        in_img=in_img.squeeze(0).cpu().numpy()
                        in_img = inpaint.inpaint_biharmonic(in_img, imgneer, channel_axis=0)
                        in_img=torch.from_numpy(in_img).cuda().unsqueeze(0)
                    #Earlier fast version: inpaint by replacing suspicious regions with mean value of non-important pixels!
                    elif args.inpaint=='mean':
                        p_neg=np.where(imgneer<=0.0)
                        in_img[:, :, [p[0]], [p[1]]] = torch.mean(in_img[:, :, [p_neg[0]], [p_neg[1]]])
                    else:
                        print("No valid inapinting method chosen")
                    boxes2, feature_map  = do_detect(model, in_img, 0.4, 0.4, True, p=None, direct_cuda_img=True)
                    sd_boxes=sd_boxes+boxes2

                #update condition (implicit):
                sd_boxes=nms(sd_boxes, 0.4, match_class=False)
                sd_boxes=nms(sd_boxes+adv_boxes, 0.4, match_class=True)
                if args.performance:
                    end=time.process_time()
                    perf_array.append(end-start-klust)
                    perf_array_cluz.append(klust)

                sdb=[]
                for box in sd_boxes:
                    sdb.append([b.detach() for b in box])
                best_arr=[]
                for cb in cbb:
                    bestest=best_iou(sdb, cb, match_class=args.eval_class)
                    best_arr.append(bestest)
                suc_atk=False
                for b in best_arr:
                    if b<iou_thresh:
                        suc_atk=True
                        break
                if args.save_outcomes:
                    clean_map=np.array(cbb)
                    clean_map[:,4]=clean_map[:,-1]
                    clean_map[:,5:]=0
                    if not len(sdb):
                        rec_map=np.zeros((1,7))
                    else:
                        rec_map=np.array(sdb)
                    rec_map[:,5]=rec_map[:,4]
                    rec_map[:,4]=rec_map[:,6]
                    rec_map=rec_map[:,:6]
                    outcome_array.append([clean_map, rec_map])
                if not suc_atk:

                    clean_corr=clean_corr+1
                else:
                    success_atk = success_atk + 1


            else:
                best_arr=[]
                for i in range(len(clean_boxes)):
                    ious['clean'].append(best_iou(cbb, [T.detach() for T in clean_boxes[i]]))
                    best=best_iou(adb, [T.detach() for T in clean_boxes[i]], args.eval_class)
                    best_arr.append(best)
                    ious['adversarial'].append(best)
                if args.save_outcomes:
                    clean_map=np.array(cbb)
                    clean_map[:,4]=clean_map[:,-1]
                    clean_map[:,5:]=0
                    if not len(adb):
                        rec_map=np.zeros((1,7))
                    else:
                        rec_map=np.array(adb)
                    rec_map[:,5]=rec_map[:,4]
                    rec_map[:,4]=rec_map[:,6]
                    rec_map=rec_map[:,:6]
                    outcome_array.append([clean_map, rec_map])

                suc_atk=False
                for b in best_arr:
                    if b<iou_thresh:#clean_pred==labels[i]:
                        suc_atk=True
                        success_atk = success_atk + 1
                        if args.geteff:
                            all_atk.append(nameee)
                        break
                if not suc_atk:
                    clean_corr=clean_corr+1
elif args.dataset in ['cifar', 'imagenet']:
    for imgfile in tqdm(val_dataset):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png') or imgfile.endswith('.JPEG'):
            nameee=imgfile.split('.')[0]
            if (eff_files != None and nameee not in eff_files and not args.uneffective) or (eff_files != None and args.uneffective and nameee in eff_files):
                continue
            kount=kount+1
            if args.ground_truth is not None:
                labelfile=os.path.abspath(os.path.join(args.ground_truth, imgfile.split('.')[0] + '.npy'))
            cheatfile = os.path.abspath(os.path.join(imgdir, imgfile))
            data = Image.open(cheatfile).convert("RGB")
            data = ds_transforms(data).cuda()
            cheat = data.unsqueeze(0).clone().detach()
            if args.clean:
                imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
                data = Image.open(imgfile).convert("RGB")
                data = ds_transforms(data).cuda()
            else:
                imgfile = os.path.abspath(os.path.join(patchdir, imgfile.split('.')[0] + '.png'))
                data = Image.open(imgfile).convert("RGB")
                data = ds_transforms_patch(data).cuda()

            gt_mask=np.zeros((data.shape[1], data.shape[2]))
            data = data.unsqueeze(0)
            if args.ground_truth is None:
                output_clean, feature_map = model(cheat)
                output_clean, feature_map = output_clean.detach().cpu().numpy()[0], feature_map.detach().cpu().numpy()[0]
                global_feature = np.mean(output_clean, axis=(0,1))
                pred_list = np.argsort(global_feature,kind='stable')
                label = pred_list[-1]
            else:
                label=np.load(labelfile)

            output_clean, feature_map = model(data)
            output_clean, feature_map = output_clean.detach().cpu().numpy()[0], feature_map.detach().cpu().numpy()[0]
            """
            DETECTION STAGE
            """
            if args.bypass_det:
                condition=True
            else:
                if args.performance_det:
                    start=time.process_time()
                fm=np.sum(feature_map,axis=0)
                biginfo=[]
                vector_s=[]
                for beta in [0.0+x*0.01 for x in range(0,100,args.ensemble_step)]:
                    biggie, clus_feats=beta_iteration(beta, fm, raw=False)
                    vector_s.append(clus_feats)
                    biginfo.append(biggie)

                #input(len(biginfo))
                if args.performance_det:
                    end=time.process_time()
                    perf_array_clus.append(end-start)
                vector_s=np.array(vector_s).reshape((1, len(vector_s), len(clus_feats)))
                detector_input = 2*nn.functional.normalize(torch.Tensor(clustering_data_preprocessing(vector_s, skip=True)), dim=2, p=float('inf')) - 1

                with torch.no_grad():
                    detector_output=net(detector_input.to(device))
                detection_score=detector_output.detach().cpu().numpy()
                condition=detection_score>=args.nn_det_threshold

                if args.performance_det:
                    end=time.process_time()
                    perf_array_det.append(end-start)

                if args.save_scores:
                    score_array.append(detection_score)
                if condition and args.bypass:
                    detected=detected+1
            """
            RECOVERY STAGE
            """
            if condition and not args.bypass:
                if args.performance:
                    klust=0
                    start=time.process_time()
                my_mask=np.zeros(gt_mask.shape)
                detected=detected+1
                global_feature = np.mean(output_clean, axis=(0,1))
                pred_list = np.argsort(global_feature,kind='stable')
                og_clean_pred = pred_list[-1]
                all_feats= np.zeros(global_feature.shape).reshape(1,-1)
                if not args.bypass_det:
                    revran_det=[0.0+x*0.01 for x in range(0,100,args.ensemble_step)]
                else:
                    if args.performance:
                        sk=time.process_time()
                    fm=np.sum(feature_map,axis=0)
                    revran_det=[]
                    if args.performance:
                        klust=klust + time.process_time()-sk
                revran=list(reversed([0.0+x*0.01 for x in range(0,100,args.inpainting_step)]))[:-1]
                corrected=False
                fm_og=fm
                bfm_old=np.ones(fm.shape)
                for numel, beta_big in enumerate(revran):
                    if beta_big in revran_det:
                        idx=revran_det.index(beta_big)
                        this=biginfo[idx]
                    else:
                        if args.performance:
                            sk=time.process_time()
                        this=beta_iteration(beta_big, fm_og)
                        if args.performance:
                            klust=klust+time.process_time()-sk

                    #no clusters to inpaint
                    if not(len(this)):
                        continue

                    dirso=[]
                    masking=0
                    for d in this:
                        if len(d)<4:
                            continue
                        dirso.append(d)
                        masking=masking+len(d)
                     #no valid cluster to inpaint (clusters must have at least 4 elements)
                    if not len(dirso):
                        continue
                    bfm=fm_og>=np.max(fm_og)*beta_big

                    #stopping condition number 1:
                    if masking > (args.neulim*my_mask.size)/81:#based on resnet50's receptive field
                        break

                    imgneer=np.zeros(my_mask.shape)
                    intereer=np.zeros((int(my_mask.shape[0]/2),int(my_mask.shape[1]/2)))

                    for mi in range(len(dirso)):
                        for x,y in dirso[mi]:
                            x,y = int(x), int(y)
                            intereer[max(2*x-1,0):min(2*x+1,intereer.shape[0])+1, max(2*y-1,0):min(2*y+1,intereer.shape[1])+1]=bfm_old[x,y]
                    p_int=np.where(intereer==1.0)
                    for x,y in zip(p_int[0], p_int[1]):
                        imgneer[max(2*x-3,0):min(2*x+3,imgneer.shape[0])+1, max(2*y-3,0):min(2*y+3,imgneer.shape[1])+1]=1.0

                    p=np.where(imgneer>0.0)
                    in_img=data.clone()
                    bfm_old=bfm

                    #saliuitl*
                    if args.inpaint=='oracle':
                        in_img[:, :, [p[0]], [p[1]]] = cheat[:, :, [p[0]], [p[1]]]
                    #saliuitl-z
                    elif args.inpaint=='zero':
                        in_img[:, :, [p[0]], [p[1]]] = 0.0
                    #Saliuitl
                    elif args.inpaint=='biharmonic':
                        in_img=denorm(in_img.squeeze(0)).cpu().numpy()#.astype('uint8')
                        in_img = inpaint.inpaint_biharmonic(in_img, imgneer, channel_axis=0)
                        in_img=ds_transforms_inp(torch.from_numpy(in_img)).cuda().unsqueeze(0)
                    #Earlier fast version: inpaint by replacing suspicious regions with mean value of non-important pixels!
                    elif args.inpaint=='mean':
                        p_neg=np.where(imgneer<=0.0)
                        in_img[:, :, [p[0]], [p[1]]] = torch.mean(in_img[:, :, [p_neg[0]], [p_neg[1]]])
                    else:
                        print("No valid inapinting method chosen")

                    output_clean2, feature_map = model(in_img, p=None)
                    output_clean2, feature_map = output_clean2.detach().cpu().numpy()[0], feature_map.detach().cpu().numpy()[0]

                    global_feature = np.mean(output_clean2, axis=(0,1))
                    pred_list = np.argsort(global_feature,kind='stable')
                    clean_pred = pred_list[-1]
                    #update condition AND stopping condition number 2 (skipped when measuring performance):
                    if clean_pred!=og_clean_pred and not args.performance:
                        corrected=True
                        break

                if args.bypass or not corrected:
                    clean_pred=og_clean_pred
                if args.performance:
                    end=time.process_time()
                    perf_array.append(end-start-klust)
                    perf_array_cluz.append(klust)
                if args.save_outcomes:
                    outcome_array.append(clean_pred==label)
                if clean_pred==label:
                    clean_corr = clean_corr + 1
                else:
                    success_atk = success_atk + 1
            else:
                global_feature = np.mean(output_clean, axis=(0,1))
                pred_list = np.argsort(global_feature,kind='stable')
                clean_pred = pred_list[-1]
                if clean_pred==label:
                    clean_corr = clean_corr + 1
                else:
                    success_atk = success_atk + 1
                    if args.geteff:
                        all_atk.append(nameee)

torch.cuda.empty_cache()
if args.save_scores:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_ens_' + str(args.ensemble_step) + '_scores'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(score_array))

if args.save_outcomes:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_inp_' + str(args.inpainting_step) + '_scores'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(outcome_array))

if args.performance:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_inp_' + str(args.inpainting_step) + '_perfs'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(perf_array))

    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_inp_' + str(args.inpainting_step) + '_clusperfs'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(perf_array_cluz))

if args.performance_det:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_ens_' + str(args.ensemble_step) + '_perfs'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(perf_array_det))

    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_ens_' + str(args.ensemble_step) + '_clusperfs'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(perf_array_clus))

if args.geteff:
    deer=os.path.join(args.patch_imgdir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + 'effective'+ '_' + str(args.n_patches) + 'p'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(all_atk))

line1="Unsuccesful Attacks:"+str(clean_corr/max(1,kount))
line2="Detected Attacks:" + str(detected/max(1,kount))
line3="Successful Attacks:" + str(success_atk/max(1,kount))
print(line1)
print(line2)
print(line3)

print("------------------------------")
#print(lines)
if args.save:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    txtpath = deer + args.det_net + '_npatches_' + str(args.n_patches) + '_det_thr_' + str(args.nn_det_threshold) + '_inp_' + str(args.inpainting_step)
    if args.clean:
        txtpath = txtpath + '_clean'
    with open(txtpath + '.txt', 'w+') as f:
        f.write('\n'.join([line1, line2, line3, "------------------------------"]))
