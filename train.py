
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import RefineDetMultiBoxLoss
from layers.modules import RefineDetMultiBoxLoss1
from layers.modules import RefineDetMultiBoxLoss2
from layers.modules import RefineDetMultiBoxLoss3
from layers.modules import RefineDetMultiBoxLoss4

from models.refinedet import build_refinedet
from models.refinedet_mish import build_refinedet_mish
from models.refinedet_swish import build_refinedet_swish
from models.refinedet_cbam import build_refinedet_cbam
from models.refinedet_sam import build_refinedet_sam
from models.refinedet_cam import build_refinedet_cam
from models.refinedet_se import build_refinedet_se
from models.refinedet_novel import build_refinedet_novel
from models.refinedet_novel_cam import build_refinedet_novel_cam
from models.refinedet_novel_sam import build_refinedet_novel_sam
from models.refinedet_novel_cbm_mish import build_refinedet_novel_cam_mish
from models.refinedet_novel_cam_swish import build_refinedet_novel_cam_swish
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.logging import Logger
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--input_size', default='320', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument("--labelSmoothing", default=False, type=str2bool, help="use label smoothing or not")
parser.add_argument("--cosineAnnealLR", default=True, type=str2bool, help="use CosineAnnealLR or not")
parser.add_argument("--CIoULoss", default=False, type=str2bool, help="use CIoULoss or not")
parser.add_argument("--DIoULoss", default=False, type=str2bool, help="use DIoULoss or not")
parser.add_argument("--GIoULoss", default=False, type=str2bool, help="use GIoULoss or not")

# The following configuration has only one True and all others are False
parser.add_argument("--useMish", default=False,type=str2bool, help="use Mish Activation function or not")
parser.add_argument("--useSwish", default=False,type=str2bool, help="use Swish Activation function or not")
parser.add_argument("--addCbam", default=False,type=str2bool, help="add CBAM block or not")
parser.add_argument("--addSam", default=False,type=str2bool, help="add SAM block or not")
parser.add_argument("--addCam", default=False,type=str2bool, help="add CAM block or not")
parser.add_argument("--addSE", default=False,type=str2bool, help="add SE block or not")
parser.add_argument("--addBA_TCB", default=False,type=str2bool, help="add Bottom-top augmentation TCB or not")
parser.add_argument("--addBA_TCB_CAM", default=True,type=str2bool, help="add BA-TCB and CAM block or not")
parser.add_argument("--addBA_TCB_SAM", default=False,type=str2bool, help="add BA-TCB and SAM block or not")
parser.add_argument("--addBA_TCB_CAM_Mish", default=False,type=str2bool, help="add BA-TCB, Swish and CAM block or not")
parser.add_argument("--addBA_TCB_CAM_Swish", default=False,type=str2bool, help="add BA-TCB, Swish and CAM block or not")
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

def train():
    if args.dataset == 'COCO':
        '''if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))'''
    elif args.dataset == 'VOC':
        '''if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')'''
        cfg = voc_refinedet[args.input_size]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    if args.addCbam:
        print("add CBAM")
        refinedet_net=build_refinedet_cbam('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addSam:
        print("add SAM")
        refinedet_net=build_refinedet_sam('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addSE:
        print("add SE")
        refinedet_net = build_refinedet_se('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addBA_TCB:
        print("add BA-TCB")
        refinedet_net = build_refinedet_novel('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addBA_TCB_CAM_Mish:
        print("add BA-TCB_cam_mish")
        refinedet_net = build_refinedet_novel_cam_mish('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addBA_TCB_CAM_Swish:
        print("add BA-TCB_cam_swish")
        refinedet_net = build_refinedet_novel_cam_swish('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addBA_TCB_CAM:
        print("add BA-TCB_cam")
        refinedet_net = build_refinedet_novel_cam('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addBA_TCB_SAM:
        print("add BA-TCB_sam")
        refinedet_net = build_refinedet_novel_sam('train', cfg['min_dim'], cfg['num_classes'])
    elif args.addCam:
        print("add CAM")
        refinedet_net=build_refinedet_cam('train', cfg['min_dim'], cfg['num_classes'])
    elif args.useMish:
        print("using Mish activity")
        refinedet_net=build_refinedet_mish('train', cfg['min_dim'], cfg['num_classes'])
    elif args.useSwish:
        print("using Swish activity")
        refinedet_net = build_refinedet_swish('train', cfg['min_dim'], cfg['num_classes'])
    else:
        print("using ReLU")
        refinedet_net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'])
    net = refinedet_net
    print(net)
    #input()

    if args.cuda:
        net = torch.nn.DataParallel(refinedet_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:

        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        refinedet_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet_net.extras.apply(weights_init)
        refinedet_net.arm_loc.apply(weights_init)
        refinedet_net.arm_conf.apply(weights_init)
        refinedet_net.odm_loc.apply(weights_init)
        refinedet_net.odm_conf.apply(weights_init)
        #refinedet_net.tcb.apply(weights_init)
        refinedet_net.tcb0.apply(weights_init)
        refinedet_net.tcb1.apply(weights_init)
        refinedet_net.tcb2.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.labelSmoothing:
        print("Using Label Smoothing")
        arm_criterion = RefineDetMultiBoxLoss1(2, 0.5, True, 0, True, 3, 0.5,
                                              False, args.cuda)
        odm_criterion = RefineDetMultiBoxLoss1(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                              False, args.cuda, use_ARM=True)
    elif args.GIoULoss:
        print("Using GIoULoss")
        arm_criterion = RefineDetMultiBoxLoss2(2, 0.5, True, 0, True, 3, 0.5,
                                               False, args.cuda)
        odm_criterion = RefineDetMultiBoxLoss2(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                               False, args.cuda, use_ARM=True)
    elif args.CIoULoss:
        print("Using CIoULoss")
        arm_criterion = RefineDetMultiBoxLoss4(2, 0.5, True, 0, True, 3, 0.5,
                                              False, args.cuda)
        odm_criterion = RefineDetMultiBoxLoss4(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                              False, args.cuda, use_ARM=True)
    elif args.DIoULoss:
        print("Using DIoULoss")
        arm_criterion = RefineDetMultiBoxLoss3(2, 0.5, True, 0, True, 3, 0.5,
                                               False, args.cuda)
        odm_criterion = RefineDetMultiBoxLoss3(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                               False, args.cuda, use_ARM=True)

    else:
        print("Using L1 smooth loss")
        arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                                              False, args.cuda)
        odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                              False, args.cuda, use_ARM=True)


    net.train()
    # loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    loss_  =0
    epoch = 0
    best_loss=100
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print(len(dataset))
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'RefineDet.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    # lr_scheduler = CosineWarmupLr(optimizer, epoch_size, ,base_lr=args.lr,warmup_epochs=10)
    scheduler = None
    if args.cosineAnnealLR:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120000, eta_min=0, last_epoch=-1)


    pltx=[]
    plty=[]
    pltz=[]
    pltm=[]

    for iteration in range(args.start_iter, cfg['max_iter']):

        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, arm_loc_loss, arm_conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            arm_loc_loss = 0
            arm_conf_loss = 0
            odm_loc_loss = 0
            odm_conf_loss = 0
            loss_= 0
            epoch += 1

        if args.cosineAnnealLR:
            pass
        else:
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        arm_loss_l, arm_loss_c = arm_criterion(out, targets)
        odm_loss_l, odm_loss_c = odm_criterion(out, targets)
        #input()
        arm_loss = arm_loss_l + arm_loss_c
        odm_loss = odm_loss_l + odm_loss_c
        loss = arm_loss + odm_loss
        loss.backward()
        optimizer.step()
        if args.cosineAnnealLR:
            scheduler.step()
        t1 = time.time()
        loss_ += loss.item()
        arm_loc_loss += arm_loss_l.item()
        arm_conf_loss += arm_loss_c.item()
        odm_loc_loss += odm_loss_l.item()
        odm_conf_loss += odm_loss_c.item()
        for param_group in optimizer.param_groups:
            out_lr=param_group['lr']

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' ||LR:%.8f Total Loss:%.4f ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f ||' \
            % (out_lr,loss.item(),arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, arm_loss_l.data[0], arm_loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(refinedet_net.state_dict(), args.save_folder
                    + '/RefineDet{}_TILDA_{}_.pth'.format(args.input_size,
            repr(iteration)))

        if iteration%500==0:
            pltx.append(iteration)
            plty.append(loss.item())
            pltz.append(arm_loss_c.item()+arm_loss_l+arm_loss_l.item())
            pltm.append(odm_loss_c.item()+odm_loss_l.item())
        # if loss<best_loss:
        #     best_loss=loss
        #     torch.save(refinedet_net.state_dict(), args.save_folder
        #              + '/RefineDet{}_{}_TILDA_sam_best_.pth'.format(args.input_size, args.dataset))
    torch.save(refinedet_net.state_dict(), args.save_folder
            + '/RefineDet{}_TILDA_final120000.pth'.format(args.input_size))
    plt.plot(pltx,plty,label='Total loss')
    plt.plot(pltx,pltz,label='CA-ARM loss')
    plt.plot(pltx,pltm,label='ODM loss')
    plt.xlabel("iteraton")
    plt.ylabel('total loss')
    plt.legend()
    plt.savefig('TILDA loss.jpg')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    train()
