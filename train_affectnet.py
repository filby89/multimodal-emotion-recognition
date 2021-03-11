import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from tsn_pytorch.transforms import *
from logger import setup_logging
from torch.utils.data.sampler import WeightedRandomSampler
from affectnet.dataset import AffectNet
import affectnet.model as module_arch
from trainer.trainer_affectnet import Trainer
import torchvision.transforms as transforms

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# calculates the weights for doing balanced sampling
def get_sampler_weights(dataset, num_classes, ages=True):
    print("Calculating sampler weights...")
    # labels_array = [x['emotion'] for x in dataset.data]
    labels_array = dataset.df.expression
    ages_array = dataset.df.age.values

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)


    sampler_weights = torch.zeros(len(labels_array))
    i=0
    for label in labels_array:
        sampler_weights[i] = class_weights[int(label)]
     
        i+=1


    return sampler_weights



def main(args, config):

    model = config.init_obj('arch', module_arch)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop(224),
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])


    dataset = AffectNet("/gpu-data/filby/affectnet_manually/",  
        "/gpu-data/filby/affectnet_manually/Manually_Annotated_file_lists/training.csv", transform=transform, age_range=None)


    weights = get_sampler_weights(dataset, 8, ages=True)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.double(),
                                                                   num_samples=len(dataset))

    def my_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


    print("Ended")

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'], shuffle=False,
        num_workers=4, pin_memory=True, sampler=train_sampler, collate_fn=my_collate)



    val_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])


    val_dataset = AffectNet("/gpu-data/filby/affectnet_manually/",  
        "/gpu-data/filby/affectnet_manually/Manually_Annotated_file_lists/validation.csv", transform=transform, age_range=None)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size'], shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=my_collate)

    logger = config.get_logger('train')
    logger.info(model)

    # get function handles of loss and metrics
    criterion_categorical = getattr(module_loss, config['loss'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion_categorical, metrics, optimizer,
                      config=config,
                      data_loader=train_loader,
                      valid_data_loader=val_loader,
                      lr_scheduler=lr_scheduler)

    trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')


    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;batch_size'),
        CustomArgs(['--embed'], type=bool, target='arch;args;embed')

    ]
    config = ConfigParser.from_args(parser, options)

    args = parser.parse_args()

    main(args, config)
