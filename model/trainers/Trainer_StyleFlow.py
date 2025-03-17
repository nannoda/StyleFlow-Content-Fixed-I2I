import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.network.net as net

from torchvision.utils import save_image
from model.network.glow import Glow
from model.utils.device import TORCH_DEV, USE_COLAB_TPU
from model.utils.utils import IterLRScheduler, remove_prefix
from tensorboardX import SummaryWriter
from model.layers.activation_norm import calc_mean_std
from model.losses.tv_loss import TVLoss

# TPU Support
# Set device
torch_device = TORCH_DEV

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    if USE_COLAB_TPU:
        import torch_xla.core.xla_model as xm
        xm.save(state, filename+'.pth.tar')  # TPU optimized saving
    else:
        torch.save(state, filename+'.pth.tar')

class merge_model(nn.Module):
    def __init__(self, cfg):
        super(merge_model, self).__init__()
        self.glow = Glow(3, cfg['n_flow'], cfg['n_block'], affine=cfg['affine'], conv_lu=not cfg['no_lu']).to(torch_device)

    def forward(self, content_images, domain_class):
        z_c = self.glow(content_images, forward=True)
        stylized = self.glow(z_c, forward=False, style=domain_class)
        return stylized

def get_smooth(I, direction):
    weights = torch.tensor([[0., 0.], [-1., 1.]], device=torch_device)
    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = weights_x.transpose(0, 1)
    weights = weights_x if direction == 'x' else weights_y

    output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))
    return output

def avg(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    gradients_I_x = get_smooth(I_gray, 'x')
    gradients_I_y = get_smooth(I_gray, 'y')

    return torch.mean(gradients_I_x * torch.exp(-10 * avg(R_gray, 'x')) + gradients_I_y * torch.exp(-10 * avg(R_gray, 'y')))
    
class Trainer():
    def __init__(self, cfg, seed=0):
        self.init = True
        set_random_seed(seed)
        self.cfg = cfg
        Mmodel = merge_model(cfg)
        
        self.model = Mmodel.to(torch_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.lr_scheduler = IterLRScheduler(self.optimizer, cfg['lr_steps'], cfg['lr_mults'], last_iter=cfg['last_iter'])
        
        vgg = net.vgg
        vgg.load_state_dict(torch.load(cfg['vgg'], map_location=torch_device))
        self.encoder = net.Net(vgg, cfg['keep_ratio']).to(torch_device)
        self.tvloss = TVLoss().to(torch_device)

        self.logger = SummaryWriter(os.path.join(self.cfg['output'], self.cfg['task_name'], 'runs'))

    def train(self, batch_id, content_iter, style_iter, source_iter, target_iter, code_iter, imgA_aug, imgB_aug, imgC_aug, imgD_aug):
        content_images = content_iter.to(torch_device)
        style_images = style_iter.to(torch_device)
        target_style = style_iter.to(torch_device)

        domain_weight = torch.tensor(1, device=torch_device)

        if self.init:
            base_code = self.encoder.cat_tensor(style_images)
            self.model(content_images, domain_class=base_code)
            self.init = False
            return

        base_code = self.encoder.cat_tensor(target_style)
        stylized = self.model(content_images, domain_class=base_code)
        stylized = torch.clamp(stylized, 0, 1)

        smooth_loss = self.tvloss(stylized) if self.cfg['loss'] == 'tv_loss' else get_gradients_loss(stylized, target_style)

        loss_c, loss_s = self.encoder(content_images, style_images, stylized, domain_weight)
        loss_c, loss_s = loss_c.mean().to(torch_device), loss_s.mean().to(torch_device)

        Loss = self.cfg['content_weight'] * loss_c + self.cfg['style_weight'] * loss_s + smooth_loss

        Loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if USE_COLAB_TPU:
            import torch_xla.core.xla_model as xm
            xm.mark_step()  # TPU sync

        current_lr = self.lr_scheduler.get_lr()[0]
        self.logger.add_scalar("current_lr", current_lr, batch_id + 1)
        self.logger.add_scalar("loss_s", loss_s.item(), batch_id + 1)
        self.logger.add_scalar("smooth_loss", smooth_loss.item(), batch_id + 1)
        self.logger.add_scalar("Loss", Loss.item(), batch_id + 1)

        if batch_id % 100 == 0:
            output_name = os.path.join(self.cfg['output'], self.cfg['task_name'], 'img_save', 
                                       f"{batch_id}_{code_iter[0].cpu().numpy()[0]}.jpg")
            output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized.cpu(), target_style.cpu()), 0)
            save_image(output_images, output_name, nrow=1)

        if batch_id % 500 == 0:
            save_checkpoint({
                'step': batch_id,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.cfg['output'], self.cfg['task_name'], 'model_save', f"{batch_id}.ckpt"))
