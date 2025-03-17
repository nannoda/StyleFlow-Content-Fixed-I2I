import argparse
import os
import torch
import subprocess
from torchvision.utils import save_image
from model.utils.utils import get_config
import model.network.net as net
from model.trainers.Trainer_StyleFlow import merge_model, set_random_seed
from model.utils.dataset import get_data_loader_folder_pair
from model.utils.sampler import DistributedGivenIterationSampler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict.yaml')
    parser.add_argument('--model_path', type=str, default='./output/wikiart/model_save/187500.ckpt.pth.tar')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video for style transfer')
    parser.add_argument('--output', type=str, default='./output', help='Output directory to save the stylized video')
    opts = parser.parse_args()
    args = get_config(opts.config)
    args['model_path'] = opts.model_path
    args['video_path'] = opts.video_path
    args['output'] = opts.output
    print('job name: ', args['job_name'])
    return args

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def extract_frames(video_path, output_folder):
    """Extract frames from the input video using FFmpeg"""
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        'ffmpeg', '-i', video_path, 
        os.path.join(output_folder, 'frame_%05d.png')
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Frames extracted to: {output_folder}")

def process_frame(model, encoder, frame, output_path):
    """Apply style transfer to a single frame and save it"""
    # Define image transformation (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize input image if necessary
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standardization
    ])

    # Apply transformation to frame
    input_image = transform(frame).unsqueeze(0).cuda()

    # Apply style transfer
    base_code = encoder.cat_tensor(input_image)  # Get the code from the encoder
    stylized_image = model(input_image, domain_class=base_code)
    stylized_image = torch.clamp(stylized_image, 0, 1)  # Clamping output between [0, 1]

    # Save the output image
    save_image(stylized_image.cpu(), output_path, nrow=1)

def combine_frames_into_video(output_folder, output_video_path, frame_rate=30):
    """Combine frames back into a video using FFmpeg"""
    cmd = [
        'ffmpeg', '-framerate', str(frame_rate), '-i', 
        os.path.join(output_folder, 'frame_%05d.png'), 
        '-c:v', 'libvpx-vp9', "-pix_fmt", "yuv420p", output_video_path 
    ]
    subprocess.run(cmd)
    print(f"Video saved to: {output_video_path}")

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

    # Extract frames from the input video
    frames_folder = os.path.join(args['output'], 'frames')
    extract_frames(args['video_path'], frames_folder)

    # Process each frame and apply style transfer
    output_frames_folder = os.path.join(args['output'], 'stylized_frames')
    os.makedirs(output_frames_folder, exist_ok=True)

    # Assuming frames_folder and output_frames_folder are defined elsewhere
    for frame_name in tqdm(sorted(os.listdir(frames_folder)), desc="Processing frames", unit="frame"):
        if frame_name.endswith('.png'):
            frame_path = os.path.join(frames_folder, frame_name)
            output_path = os.path.join(output_frames_folder, frame_name)

            # Load frame
            frame = Image.open(frame_path).convert('RGB')
            process_frame(model, encoder, frame, output_path)

    # Combine frames into a new video
    output_video_path = os.path.join(args['output'], 'stylized_video.webm')
    combine_frames_into_video(output_frames_folder, output_video_path)

if __name__ == "__main__":
    main()
