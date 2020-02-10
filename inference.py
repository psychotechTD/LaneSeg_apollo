import argparse
import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import pandas as pd

from dataloaders import make_data_loader
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from dataloaders.utils_zzm import decode_color_labels, decode_segmap, decode_segmap_gray_apollo, encode_segmap
from tqdm import tqdm
from dataloaders import custom_transforms as tr
from torchvision import transforms

def makeargs():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='mobilenet',  ##########################改backbone
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='apollo',
                        choices=['pascal', 'coco', 'cityscapes', 'apollo'],
                        help='dataset name (default: pascal)')
    # parser.add_argument('--use-sbd', action='store_true', default=False,
    #                     help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args([])
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'apollo':20
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 2 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = 1

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'apollo': 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)#打印参数信息
    return args

#
# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         kwargs = {'num_workers': args.workers, 'pin_memory': True}
#         self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
#
#         model = DeepLab(num_classes=self.nclass, backbone=args.backbone,
#                         output_stride=args.output_stride, sync_bn=args.sync_bn,
#                         freeze_bn=args.freeze_bn)
#         train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
#                         {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
#         optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
#                                     weight_decay=args.weight_decay, nesterov=args.nesterov)
#         self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
#         self.model, self.optimizer = model, optimizer
#         self.evaluator = Evaluator(self.nclass)
#         self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
#
#         self.model = model.cuda()
#
#         self.best_pred = 0.0
#         if args.resume is not None:
#             if not os.path.isfile(args.resume):
#                 raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             args.start_epoch = checkpoint['epoch']
#             if args.cuda:
#                 self.model.module.load_state_dict(checkpoint['state_dict'])
#             else:
#                 self.model.load_state_dict(checkpoint['state_dict'])
#             if not args.ft:
#                 self.optimizer.load_state_dict(checkpoint['optimizer'])
#             self.best_pred = checkpoint['best_pred']
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#
#         if args.ft:
#             args.start_epoch = 0
#
#     def training(self, epoch):
#         train_loss = 0.0
#         self.model.train()
#         tbar = tqdm(self.train_loader)
#         num_img_tr = len(self.train_loader)
#         for i, sample in enumerate(tbar):
#             image, target = sample['image'], sample['label']
#             image, target = image.cuda(), target.cuda()
#             self.scheduler(self.optimizer, i, epoch, self.best_pred)
#             self.optimizer.zero_grad()
#             output = self.model(image)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()
#             train_loss += loss.item()
#             tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
#
#         print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
#         print('Loss: %.3f' % train_loss)
#
#     def validation(self, epoch):
#         self.model.eval()
#         self.evaluator.reset()
#         tbar = tqdm(self.val_loader, desc='\r')
#         test_loss = 0.0
#         for i, sample in enumerate(tbar):
#             image, target = sample['image'], sample['label']
#             image, target = image.cuda(), target.cuda()
#             with torch.no_grad():  #
#                 output = self.model(image)
#             loss = self.criterion(output, target)
#             test_loss += loss.item()
#             tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
#             pred = output.data.cpu().numpy()
#             target = target.cpu().numpy()
#             pred = np.argmax(pred, axis=1)
#             self.evaluator.add_batch(target, pred)
#
#         Acc = self.evaluator.Pixel_Accuracy()
#         mIoU = self.evaluator.Mean_Intersection_over_Union()
#         print('Validation:')
#         print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
#         print("Acc:{}, mIoU:{},".format(Acc, mIoU))
#         print('Loss: %.3f' % test_loss)


def test(model_path):
    args = makeargs()
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    print('Loading model...')
    model = DeepLab(num_classes=8, backbone='drn', output_stride=args.output_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model.eval()
    checkpoint = torch.load(model_path)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print('Done')
    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
    evaluator = Evaluator(nclass)
    evaluator.reset()

    print('Model infering')
    test_dir = 'test_example1'
    test_loss = 0.0
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()

        with torch.no_grad():  #
            output = model(image)
        loss = criterion(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)


    print(image.shape)
    Acc = evaluator.Pixel_Accuracy()
    mIoU = evaluator.Mean_Intersection_over_Union()
    print('testing:')
    print("Acc:{}, mIoU:{},".format(Acc, mIoU))
    print('Loss: %.3f' % test_loss)




#解构inference过程
IMAGE_SIZE = [(1024, 384), (1536,512), (3384,1020), (2048, 768), (1636, 640)]


def transform(sample, cropsize=IMAGE_SIZE[1], offset=690):#####################此处更改IMAGE_SIZE选择
    roi_image = sample['image'].crop((0,offset,3384,1710))
    roi_label = sample['label'].crop((0,offset,3384,1710))
    sample['image'] = roi_image.resize(cropsize, Image.BILINEAR)
    sample['label'] = roi_label.resize(cropsize, Image.NEAREST)
    composed_transform  = transforms.Compose([
#         tr.RandomHorizontalFlip(),
#         tr.RandomGaussianBlur(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    return composed_transform(sample)

def expand_resize_data(prediction=None, submission_size=[3384, 1710], offset=690):
    expand_mask = cv2.resize(prediction, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask

def load_model(model_path):
    args = makeargs()
    print('Loading model...')
    model = DeepLab(num_classes=8,backbone=args.backbone, output_stride=args.output_stride,#改backbone
                   sync_bn=args.sync_bn,freeze_bn=args.freeze_bn)
    model.eval()
    checkpoint = torch.load(model_path)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print('model loaded')
    return model

def unziptest(model_path):
    model = load_model(model_path)
    sub_dir = './test_example/test_example_size1_69epoch_color/'
    Test_dir = pd.read_csv("data_list/test_lite.csv")
    test_dir = Test_dir["image"].values
    test_dir_lb = Test_dir["label"].values
    print("start inference...")

    for i in range(0, len(test_dir)):
        image = cv2.imread(test_dir[i])
        target = cv2.imread(test_dir_lb[i])
        _img = Image.fromarray(image)
        _target = Image.fromarray(target)
        sample = {'image': _img, 'label': _target}
        _img, _target = transform(sample)['image'], transform(sample)['label']
        _img = _img.unsqueeze(dim=0)
        _img = _img.cuda()

        with torch.no_grad():
            pred = model(_img)
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = torch.squeeze(pred)
        pred = pred.detach().cpu().numpy()
        print("inference done %d of %d" % (i, len(test_dir)))
        prediction = expand_resize_data(pred)
        color_labels = decode_color_labels(prediction)
        color_labels = np.transpose(color_labels, (1, 2, 0))
        # img_ori = cv2.imread(test_dir[i])
        # added = cv2.addWeighted(img_ori, 1, color_labels, 0.3, 0)
        cv2.imwrite(os.path.join(sub_dir, test_dir[i][-29:]), color_labels)

if __name__ == "__main__":
    #快速inference
    model_path = 'run/apollo/deeplab-mobilenet/size1_69epoch/checkpoint.pth.tar'
    # test(model_path)
    unziptest(model_path)