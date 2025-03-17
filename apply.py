import argparse
import os

import torch
from torchvision.utils import save_image
from model.utils.utils import get_config
import model.network.net as net
from model.trainers.Trainer_StyleFlow import merge_model, set_random_seed
from model.utils.dataset import get_data_loader_folder_pair
from model.utils.sampler import DistributedGivenIterationSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict.yaml')
    parser.add_argument('--model_path', type=str, default='./output/wikiart/model_save/187500.ckpt.pth.tar')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image for style transfer')
    parser.add_argument('--output', type=str, default='./output', help='Output directory to save the stylized image')
    opts = parser.parse_args()
    args = get_config(opts.config)
    args['model_path'] = opts.model_path
    args['image_path'] = opts.image_path
    args['output'] = opts.output
    print('job name: ', args['job_name'])
    return args

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def main():
    torch.backends.cudnn.benchmark = True
    set_random_seed(0)

    args = parse_args()

    # Create output directory if not exists
    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
        print('mkdir args.output')

    # Load the trained model
    model = merge_model(args)
    if os.path.isfile(args['model_path']):
        print("--------loading checkpoint----------")
        checkpoint = torch.load(args['model_path'])
        checkpoint['state_dict'] = remove_prefix(checkpoint['state_dict'], 'module.')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args['model_path']))
    else:
        raise('no checkpoint found', args['model_path'])

    # Load VGG and encoder model
    vgg = net.vgg
    vgg.load_state_dict(torch.load(args['vgg']))
    encoder = net.Net(vgg).cuda()

    # Move model to GPU and set it to evaluation mode
    model.cuda()
    model.eval()

    # Load the input image
    from torchvision import transforms
    from PIL import Image

    # Define image transformation (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize input image if necessary
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standardization
    ])

    # Open the image and apply the transformation
    input_image = Image.open(args['image_path']).convert('RGB')
    input_image = transform(input_image).unsqueeze(0).cuda()

    # Apply style transfer
    base_code = encoder.cat_tensor(input_image)  # Get the code from the encoder
    stylized_image = model(input_image, domain_class=base_code)
    stylized_image = torch.clamp(stylized_image, 0, 1)  # Clamping output between [0, 1]

    # Save the output image
    output_name = os.path.join(args['output'], 'stylized_image.png')
    save_image(stylized_image.cpu(), output_name, nrow=1)
    print(f"Stylized image saved to: {output_name}")

if __name__ == "__main__":
    main()
