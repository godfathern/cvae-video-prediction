import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD

# Loss = MSE + β × KL
class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.current_epoch = current_epoch
        if args.kl_anneal_type == 'Cyclical':
            self.schedule = self.frange_cycle_linear(args.num_epoch, n_cycle=args.kl_anneal_cycle, ratio=args.kl_anneal_ratio)
        elif args.kl_anneal_type == 'Monotonic':
            self.schedule = self.frange_cycle_linear(args.num_epoch, n_cycle=1, ratio=args.kl_anneal_ratio)
        else:  # 'None'
            self.schedule = np.ones(args.num_epoch)
        
    def update(self):
        # TODO
        self.current_epoch += 1
    
    def get_beta(self):
        # TODO
        epoch = min(self.current_epoch, len(self.schedule) - 1 )
        return float(self.schedule[epoch])
    
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and int(i + c * period) < n_iter:
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.lr_milestones, gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        # Start a fresh log for a new run; only append when resuming from a checkpoint
        log_path = os.path.join(self.args.save_root, "train_log.csv")
        if self.args.ckpt_path is None or not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("epoch,train_loss,beta,tfr,teacher_forcing_on,val_mse,val_psnr\n")

        for i in range(self.args.num_epoch):
            self.train()                                    # BatchNorm uses batch statistics while training
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            epoch_loss, n_batches = 0.0, 0
            beta = self.kl_annealing.get_beta()
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_loss += float(loss.detach().cpu()); n_batches += 1
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            val_mse = self.eval()
            with open(log_path, "a") as f:
                f.write(f"{self.current_epoch},{epoch_loss / max(n_batches, 1):.6f},{beta:.4f},{self.tfr:.3f},"
                        f"{int(adapt_TeacherForcing)},{val_mse:.6f},{self.last_val_psnr:.3f}\n")
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        # This method shadows nn.Module.eval, so switch BatchNorm to its running
        # statistics explicitly and back to train mode when done. (Measured: same
        # PSNR as batch statistics within 0.03 dB, but no running-stat pollution.)
        nn.Module.eval(self)
        val_loader = self.val_dataloader()
        loss = torch.tensor(0.0)
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        self.train()
        return float(loss.detach().cpu())
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        # ---------------------------------------------------------------
        # Shape of img and label: (B, T, C, H, W) = (2, 16, 3, 32, 64)
        #
        #   axis 0  B = 2   which clip in the batch (DataLoader stacks clips here)
        #   axis 1  T = 16  which frame in the clip  -> this is TIME
        #   axis 2  C = 3   colour channel: red, green, blue
        #   axis 3  H = 32  pixel row
        #   axis 4  W = 64  pixel column
        T = img.shape[1]
        mse_total = 0.0
        kl_total = 0.0
        prev_feat = self.frame_transformation(img[:, 0])   # frame 0 is given (x1 in the figure)
        
        for t in range(1, T):
            target = img[:, t]
            pose = label[:, t]

            # Lab 4 - instructions : figure 4a
            # Right box : the encoder peeks at the answer
            target_feat = self.frame_transformation(target)
            pose_feat = self.label_transformation(pose)
            z, mu, logvar = self.Gaussian_Predictor(target_feat, pose_feat)
            
            # Left box: the decoder draws the frame from the previous frame's features
            fused = self.Decoder_Fusion(prev_feat, pose_feat, z)
            pred = self.Generator(fused)

            # The two losses
            mse_total += self.mse_criterion(pred, target)
            kl_total += kl_criterion(mu, logvar, self.batch_size)

            # Teacher forcing: what the next step builds on.
            # ON  -> the real frame t, whose features were just computed (no second encoder pass)
            # OFF -> our own prediction, clamped to [0, 1] exactly as at inference time
            if adapt_TeacherForcing:
                prev_feat = target_feat
            else:
                prev_feat = self.frame_transformation(pred.clamp(0.0, 1.0))

        beta = self.kl_annealing.get_beta()
        loss = mse_total + beta * kl_total

        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
        return loss
                
    
    def val_one_step(self, img, label):
        # Lab 4 - instructions : section 3.3 / figure 5 (inference)
        # img, label: (1, 630, 3, 32, 64). Only frame 0 is used as input;
        # every later frame is built on the model's own previous prediction.
        T = img.shape[1]
        prev_frame = img[:, 0]
        mse_total = 0.0
        psnr_list = []
        decoded_frame_list = [img[:, 0].cpu()] if self.args.store_visualization else None

        # Encode all T poses in one batched call (eval mode: BatchNorm uses running
        # statistics, so batching does not change the result)
        pose_feats = self.label_transformation(label[0])      # (T, L_dim, H, W)

        for t in range(1, T):
            target = img[:, t]
            pose_feat = pose_feats[t:t+1]
            prev_feat = self.frame_transformation(prev_frame)

            # No encoder at inference: z comes straight from the prior N(0, I)
            z = torch.randn(img.shape[0], self.args.N_dim, self.args.frame_H, self.args.frame_W,
                            device=img.device)

            fused = self.Decoder_Fusion(prev_feat, pose_feat, z)
            pred = self.Generator(fused).clamp(0.0, 1.0)

            mse_total += self.mse_criterion(pred, target)
            psnr_list.append(Generate_PSNR(pred, target).item())
            if decoded_frame_list is not None:
                decoded_frame_list.append(pred.cpu())
            prev_frame = pred                       # red dashed arrow in figure 5

        self.last_val_psnr = float(np.mean(psnr_list))
        self.last_val_psnr_per_frame = psnr_list
        print(f"\n[val] epoch {self.current_epoch}: avg PSNR over {T-1} frames = {self.last_val_psnr:.2f} dB")

        # Per-frame PSNR for the report (handout 5.1: "plot the per-frame PSNR")
        np.save(os.path.join(self.args.save_root, f"val_psnr_epoch{self.current_epoch}.npy"), np.array(psnr_list))

        if self.args.store_visualization:
            frames = stack(decoded_frame_list).squeeze(1)       # (630, 3, 32, 64)
            self.make_gif(frames, os.path.join(self.args.save_root, f"val_epoch{self.current_epoch}.gif"))

        return mse_total / (T - 1)
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # Lab 4 - instructions : section 3.2 (teacher forcing)
        # Keep tfr at its initial value until epoch tfr_sde, then lower it by
        # tfr_d_step every epoch so the model learns to build on its own guesses.
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path, map_location=self.args.device)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.args.lr_milestones, gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[2, 5], help="epochs at which the learning rate is multiplied by 0.1 (template default: 2 5)")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical', choices=['Cyclical', 'Monotonic', 'None'], help="KL annealing schedule")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
