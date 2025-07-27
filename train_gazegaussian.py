import argparse
import random
from configs.gazegaussian_options import BaseOptions
import numpy as np
import torch
from trainer.gazegaussian_trainer import GazeGaussianTrainer
from utils.recorder import GazeGaussianTrainRecorder

def auto_argparse_from_class(cls_instance):
    parser = argparse.ArgumentParser(description="Auto argparse from class")
    
    for attribute, value in vars(cls_instance).items():
        if isinstance(value, bool):
            parser.add_argument(f'--{attribute}', action='store_true' if not value else 'store_false',
                                help=f"Flag for {attribute}, default is {value}")
        elif isinstance(value, list):
            parser.add_argument(f'--{attribute}', type=type(value[0]), nargs='+', default=value,
                                help=f"List for {attribute}, default is {value}")
        else:
            parser.add_argument(f'--{attribute}', type=type(value), default=value,
                                help=f"Argument for {attribute}, default is {value}")

    return parser

def main():
    """Main function"""
    torch.manual_seed(2024)  # cpu
    torch.cuda.manual_seed(2024)  # gpu
    np.random.seed(2024)  # numpy
    random.seed(2024)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True

    base_options  = BaseOptions()
    parser = auto_argparse_from_class(base_options)
    opt = parser.parse_args()

    from dataloader.eth_xgaze import get_train_loader, get_val_loader

    train_data_loader = get_train_loader(
        opt, data_dir = opt.img_dir, batch_size = opt.batch_size, num_workers = opt.num_workers, evaluate="landmark", is_shuffle=True, dataset_name=opt.dataset_name
    )

    recorder = GazeGaussianTrainRecorder(opt)
    trainer = GazeGaussianTrainer(opt, recorder)


    trainer.train(
        train_data_loader=train_data_loader,
        n_epochs=opt.num_epochs,
        valid_data_loader=None,
    )


if __name__ == "__main__":
    main()
