import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from EmoReact.transforms import *
from logger import setup_logging
from model import loss
from EmoReact.dataset import TSNDataSet
from trainer.trainer import Trainer
from EmoReact.models import TSN

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args, config):


    if args.modality == 'RGB':
        data_length = 1
    elif args.modality == "depth":
        data_length = args.data_length
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = args.data_length


    model = TSN(8, args.num_segments, args.modality, modalities_fusion=args.modalities_fusion, num_feats=args.num_feats,
                base_model=args.arch, new_length=data_length, embed=args.embed,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn, categorical=args.categorical, continuous=args.continuous, audio=args.audio)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation()

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()


    dataset = TSNDataSet("train", num_segments=args.num_segments,
                   new_length=data_length,        
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                   GroupScale((256,256)),
                   GroupRandomHorizontalFlip(),
                   GroupRandomCrop(224),

                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))



    collate_fn = None
    sampler = None
    

    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("val", num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                    image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale((int(224),int(224))),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)


    logger = config.get_logger('train')
    logger.info(model)

    criterion_categorical = getattr(module_loss, config['loss'])
    criterion_continuous = getattr(module_loss, config['loss_continuous'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    metrics_continuous = [getattr(module_metric, met) for met in config['metrics_continuous']]


    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)


    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics, metrics_continuous, optimizer,
                      categorical=args.categorical,
                      continuous=args.continuous,
                      config=config,
                      data_loader=train_loader,
                      valid_data_loader=val_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


    test_loader = torch.utils.data.DataLoader(
        TSNDataSet("test", num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                    image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale((int(224),int(224))),
                       # GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)


    # load best model and evaluate on test
    cp = torch.load(str(trainer.checkpoint_dir / 'model_best.pth'))

    model.load_state_dict(cp['state_dict'],strict=True)
    print('loaded', str(trainer.checkpoint_dir / 'model_best.pth'), 'best_epoch', cp['epoch'])

    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics, metrics_continuous, optimizer,
                      categorical=args.categorical,
                      continuous=args.continuous,
                      config=config,
                      data_loader=train_loader,
                      valid_data_loader=test_loader,
                      lr_scheduler=lr_scheduler)
 
    trainer.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # parser.add_argument('--exp_name', type=str)

    parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'depth'])

    # ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num_segments', type=int, default=5)
    parser.add_argument('--consensus_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--data_length', type=int, default=5)

    parser.add_argument('--modalities_fusion', type=str, default='cat')
    parser.add_argument('--lossembed', type=str, default='mse')

    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')

    # ========================= Learning Configs ==========================
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                        metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
    parser.add_argument('--categorical', default=False, action="store_true")
    parser.add_argument('--continuous', default=False, action="store_true")
    parser.add_argument('--embed', default=False, action="store_true")
    parser.add_argument('--num_feats', default=2048, type=int)

    parser.add_argument('--audio', default=False, action="store_true")


    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                        metavar='N', help='evaluation frequency (default: 5)')


    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--flow_prefix', default="", type=str)


    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp_name'], type=str, target='name'),
    ]
    config = ConfigParser.from_args(parser, options)
    print(config)

    args = parser.parse_args()

    main(args, config)



#