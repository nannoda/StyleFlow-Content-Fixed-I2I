import argparse
import os
import torch
from torch.utils.data import DataLoader

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

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

    if TPU_AVAILABLE and 'COLAB_TPU_ADDR' in os.environ:
        torch_device = xm.xla_device()
        print("Using TPU")
    elif torch.cuda.is_available():
        torch_device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        torch_device = torch.device('cpu')
        print("Using CPU")

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

    for batch_id, batch in enumerate(train_loader):
        batch = [x.to(torch_device) for x in batch]  # Move tensors to TPU/CUDA
        trainer.train(batch_id, *batch)

        if TPU_AVAILABLE:
            xm.mark_step()  # Ensure TPU execution

    print("Training completed.")


if __name__ == "__main__":
    main()
