import warnings
warnings.filterwarnings('ignore')
import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from logger import setup_logging
from model import loss
from EmoReact.dataset import TSNDataSet, AudioDataSet
from trainer.trainer import Trainer
from EmoReact.models import AudioOnly
import torchvision.transforms as transforms

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args, config):

    
    model = AudioOnly(8, base_model=args.arch)
 
    import torchaudio.transforms as at

    t = []
    if args.masking_time != 0:
        t.append(at.TimeMasking(args.masking_time))

    if args.masking_freq != 0:
        t.append(at.FrequencyMasking(args.masking_freq))

    transform = transforms.Compose(t)

    dataset = AudioDataSet("train", transform=transform)

    val_transform = transforms.Compose([
       
        ])
 
    sampler = None
    

    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=None, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        AudioDataSet("val",transform=val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)


    logger = config.get_logger('train')
    logger.info(model)

    criterion_categorical = getattr(module_loss, config['loss'])
    criterion_continuous = getattr(module_loss, config['loss_continuous'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    metrics_continuous = [getattr(module_metric, met) for met in config['metrics_continuous']]

    # policies = model.get_optim_policies(lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics, metrics_continuous, optimizer,
                      categorical=True,
                      continuous=False,
                      config=config,
                      data_loader=train_loader,
                      valid_data_loader=val_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


    test_loader = torch.utils.data.DataLoader(
        AudioDataSet("test",transform=val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)
   

    """ load best model and test """
    cp = torch.load(str(trainer.checkpoint_dir / 'model_best.pth'))

    model.load_state_dict(cp['state_dict'],strict=True)
    print('loaded', str(trainer.checkpoint_dir / 'model_best.pth'), 'best_epoch', cp['epoch'])

    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics, metrics_continuous, optimizer,
                      categorical=True,
                      continuous=False,
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

    # ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--num_segments', type=int, default=5)


    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')

    # ---- transforms ---- #
    parser.add_argument('--masking_time', type=int, default=0)
    parser.add_argument('--masking_freq', type=int, default=0)

    # ========================= Learning Configs ==========================
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-3, type=float,
                        metavar='W', help='weight decay (default: 5e-3)')
    parser.add_argument('--categorical', default=True, action="store_true")
    parser.add_argument('--continuous', default=False, action="store_true")

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                        metavar='N', help='evaluation frequency (default: 5)')

    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp_name'], type=str, target='name'),
    ]
    config = ConfigParser.from_args(parser, options)
    print(config)

    args = parser.parse_args()

    main(args, config)
