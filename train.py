import argparse
import os
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar

from model.utils.device import TORCH_DEV, USE_COLAB_TPU
from model.trainers.Trainer_StyleFlow import Trainer, set_random_seed
from model.utils.dataset import get_data_loader_folder_pair
from model.utils.sampler import DistributedGivenIterationSampler
from model.utils.utils import get_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict.yaml')
    opts = parser.parse_args()
    args = get_config(opts.config)
    print('Job name: ', args['job_name'])
    return args

def main():
    torch.backends.cudnn.benchmark = True
    torch_device = TORCH_DEV
    print(f"Device: {torch_device}")
    set_random_seed(0)

    last_iter = -1
    args = parse_args()

    # Create necessary directories
    os.makedirs(os.path.join(args['output'], args['task_name'], 'img_save'), exist_ok=True)
    os.makedirs(os.path.join(args['output'], args['task_name'], 'model_save'), exist_ok=True)
    print("Directories created.")

    trainer = Trainer(args)
    train_dataset = get_data_loader_folder_pair(
        args['rootA'], args['rootB'], args['infotxt'], args['batch_size'],
        True, args['keep_percent'], get_direct=args['get_direct'],
        used_domain=args['used_domain'], train_vr=args['train_vr']
    )

    train_sampler = DistributedGivenIterationSampler(
        train_dataset, args['max_iter'], args['batch_size'], world_size=1, rank=0, last_iter=last_iter
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=False, sampler=train_sampler
    )

    for batch_id, batch in enumerate(tqdm(train_loader, desc='Training Progress', unit='batch')):
        batch = [x.to(torch_device) for x in batch]  # Move tensors to TPU/CUDA
        trainer.train(batch_id, *batch)
        if USE_COLAB_TPU:
            import torch_xla.core.xla_model as xm
            xm.mark_step()  # Ensure TPU execution

    print("Training completed.")

if __name__ == "__main__":
    main()
