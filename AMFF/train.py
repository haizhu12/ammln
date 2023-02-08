import argparse
import os
import random
import time
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import pandas as pd
import torch.nn as nn
from senet import cse_resnet50
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from loss import CrossMatchingTripletLoss, WeightedCrossMatchingTripletLoss
from model import Model
from utils import DomainDataset, compute_metric
from utils1 import load_data, AverageMeter, accuracy
from train_cse_resnet_tuberlin_ext import SoftCrossEntropy
print("torch.cuda.is_available()",torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False

# train for one epoch
def train(net, data_loader, train_optimizer, criterion_train, criterion_train_t):
    batch_time = AverageMeter()
    optimizer.zero_grad()
    end = time.time()
    net.train()
    total_loss, total_num, train_bar = 0.0, 0.0, tqdm(data_loader, dynamic_ncols=True)
    for img, domain, label, img_name in train_bar:
        #domain = domain.type(torch.LongTensor).view(-1,)
        # dom = domain
        # dom = dom.to(device)
        # domain = domain.numpy()
        # le=len(domain)
        # domain = domain.reshape(le, 1)
        # domain = torch.from_numpy(domain)
        # domain = domain.to(device)
        # img = img.to(device)
        # label=label.to(device)
        domain = domain.to(device)
        img = img.to(device)
        label=label.to(device)
        _, _, _, output_labels, proj = net(img)
        loss_ns = loss_criterion(proj, label)
        loss_cross = criterion_train(output_labels, label)

        # local metric loss
        if args.tri_lambda > 0:
            loss_t, s_ap, s_an = criterion_train_t(proj, label, domain)
        else:
            loss_t = 0 * loss_ns

        #all loss
        #loss = loss_ns + args.tri_lambda * loss_t + args.tri_lambda * loss_cross + args.tri_lambda * loss_kd

        loss = loss_ns + args.tri_lambda * loss_t + args.tri_lambda * loss_cross

        # compute gradient and do SGD step
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += img.size(0)
        total_loss += float(loss.item() * img.size(0))
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # tensorboard
        # train_loss1.append(total_loss / total_num)
        # loss_ns1.append(loss_ns)
        # loss_cross1.append(loss_cross)
        # loss_kd1.append(loss_kd)
        # loss_t1.append(loss_t)
    return total_loss / total_num

# val for one epoch
def val(net, data_loader):
    net.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label, img_name in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            _,_,_,output_labels, proj = net(img.cuda())
            vectors.append(proj.cpu())
            domains.append(domain)
            labels.append(label)
        vectors = torch.cat(vectors, dim=0)
        domains = torch.cat(domains, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = compute_metric(vectors, domains, labels)
        results['P@100'].append(acc['P@100'] * 100)
        results['P@200'].append(acc['P@200'] * 100)
        results['mAP@200'].append(acc['mAP@200'] * 100)
        results['mAP@all'].append(acc['mAP@all'] * 100)
        print('Val Epoch: [{}/{}] | P@100:{:.1f}% | P@200:{:.1f}% | mAP@200:{:.1f}% | mAP@all:{:.1f}%'
              .format(epoch, epochs, acc['P@100'] * 100, acc['P@200'] * 100, acc['mAP@200'] * 100,
                      acc['mAP@all'] * 100))
        if start_val <= epoch:
            p100.append(acc['P@100'] * 100)
            p200.append(acc['P@200'] * 100)
            map200.append(acc['mAP@200'] * 100)
            mapall.append(acc['mAP@all'] * 100)
    return acc['precise'], vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_root', default='./data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin', "sketchy_c100","quickdraw"],help= 'Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'], help='Backbone type')
    parser.add_argument('--proj_dim', default=512, type=int, help='Projected embedding dim')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--warmup', default=1, type=int, help='Number of warmups over the model to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    parser.add_argument('--num-workers', metavar='N', type=int, default=2, help='number threads to load data')
    parser.add_argument('--epoch_lenth', default=200, type=int, metavar='N', help='iterations per epoch')
    parser.add_argument('--num_instance', '-n', default=8, type=int, metavar='N', help='number of img per class')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--num_classes', metavar='N', type=int, default=220, help='number of classes (default: 220)')
    parser.add_argument('--loss', metavar='LAMBDA', default='ce', type=str, help='loss type')
    parser.add_argument('--tri_lambda', metavar='LAMBDA', default='0.0', type=float, help='lambda for triplet loss (default: 1)')
    parser.add_argument('--margin',default=0.2, type=float, help='margin for triplet loss')
    parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot', type=str, help='zeroshot version for training and testing (default: zeroshot)')
    parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float, help='lambda for kd loss (default: 1)')
    parser.add_argument('--start_val', default=20, type=int, help='start_val')
    args = parser.parse_args()
    # for ii in range(1,5):
    #     if ii == 1:
    #         args.data_name = 'sketchy_c100'
    #     elif ii == 2:
    #         args.data_name = 'sketchy'
    #     elif ii == 3:
    #         args.data_name = 'tuberlin'
    #     elif ii == 4:
    #         args.data_name = 'quickdraw'
    #         args.epochs = 3
    print("args.data_name ",args.data_name )
# args parse

    if args.loss == 'ce':
        args.tri_lambda = 0
    elif args.loss == 'cross':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin, normalize_feature=True, mode='basic')
    elif args.loss == 'within':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin, normalize_feature=True, mode='within')
    elif args.loss == 'hybrid':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin, normalize_feature=True, mode='partial')
    elif args.loss == 'all':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin, normalize_feature=True, mode='all')
    elif args.loss == 'mathm':
        criterion_train_t = WeightedCrossMatchingTripletLoss(margin=args.margin, normalize_feature=True, mode='all')
        # num_clasess
    if args.data_name == 'tuberlin':
        args.num_classes = 220
    elif args.data_name == 'sketchy':
        args.num_classes = 104
    elif args.data_name == 'sketchy_c100':
        args.num_classes = 100
    elif args.data_name == 'quickdraw':
        args.num_classes = 80

    print("args.tri_lambda",args.tri_lambda)
    #args = parser.parse_args()
    data_root, data_name, backbone_type, proj_dim = args.data_root, args.data_name, args.backbone_type, args.proj_dim
    batch_size, epochs, warmup, save_root ,start_val= args.batch_size, args.epochs, args.warmup, args.save_root,args.start_val

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    #mix_loader, __ = load_data(args)

    # model and loss setup
    model = Model(backbone_type, proj_dim, args.num_classes).cuda()
    # #teacher model
    # print(str(datetime.datetime.now()) + ' student model inited.')
    # model_t = cse_resnet50(num_classes=args.num_classes)
    # model_t = nn.DataParallel(model_t).cuda()
    # print(str(datetime.datetime.now()) + ' teacher model inited.')
    #loss_cross
    criterion_train = nn.CrossEntropyLoss()
    #loss_ns
    loss_criterion = NormalizedSoftmaxLoss(args.num_classes, proj_dim).cuda()
    # #teacher model loss
    # criterion_train_kd = SoftCrossEntropy().cuda()
    # optimizer config
    optimizer = AdamW([{'params': model.parameters()}, {'params': loss_criterion.parameters(), 'lr': 1e-1}], lr=1e-5, weight_decay=5e-4)
    # training loop
    results = {'train_loss': [], 'val_precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, proj_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #train
    best_precise = 0.0
    train_loss=0.0
    p100, p200, map200, mapall = [], [], [], []
    #train_loss1,loss_ns1, loss_cross1, loss_kd1, loss_t1 = [], [], [], [], []
    #writer=SummaryWriter("logs_tuberlin")
    for epoch in range(1, epochs + 1):
        # warmup, not update the parameters of backbone
        for param in model.backbone.parameters():
            param.requires_grad = False if epoch <= warmup else True
        train_loss= train(model, train_loader, optimizer,criterion_train,criterion_train_t)
        results['train_loss'].append(train_loss)
        if start_val <= epoch:
            val_precise, features = val(model, val_loader)
            results['val_precise'].append(val_precise * 100)
        # # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')
        # if val_precise > best_precise:
        #     best_precise = val_precise
        #     torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
        #     torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
    # for ii, x in enumerate(train_loss1):
    #     writer.add_scalar("train_loss1",x,ii)
    # for ii, x in enumerate(loss_ns1):
    #     writer.add_scalar("loss_ns1", x, ii)
    # for ii, x in enumerate(loss_cross1):
    #     writer.add_scalar("loss_cross1",x,ii)
    # for ii, x in enumerate(loss_kd1):
    #     writer.add_scalar("loss_kd1",x,ii)
    # for ii, x in enumerate(loss_t1):
    #     writer.add_scalar("loss_t1",x,ii)

    a,b,c,d=0.0,0.0,0.0,0.0
    for it in p100:
        a += it
    print("p@100",a/len(p100))
    for it in p200:
        b += it
    print("p@200",b/len(p200))
    for it in map200:
        c += it
    print("map@200",c/len(map200))
    for it in mapall:
        d += it
    print("map@all",d/len(mapall))