"""
This is based on @dome272's implementation of a conditional diffusion model on CIFAR. 

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import wandb
from utils import *
from modules import UNet_conditional, EMA

# based on stats in Nichol and Dhariwal, 2021
config = SimpleNamespace(    
    save_every = 10,
    run_name = "TestHumans",
    epochs = 81,
    noise_steps=1000,
    seed = 789,
    batch_size = 50,
    img_size = 64,
    num_classes = 6, # gets invoked as num_classes - 1 in code
    dataset_path = get_synthetic(img_size=64),
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 8, # save images to wandb
    num_workers=1,
    lr = 1e-4) # bit less than 2021 paper


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

        # Actual Labels
        self.step_map = {
            0: [3, 1, 2],
            1: [4, 1, 2], 
            2: [3, 0, 2],
            3: [4, 0, 2],
            4: [0, 2, 0],
            5: [1, 2, 1]
        }

    def prepare_noise_schedule(self):
        """
        Note that our noise schedule also matches the 2021 paper:
        "When using the linear noise schedule from Ho et al. (2020), 
        we linearly interpolate from β1 = 0.0001/4 to β4000 =
        0.02/4 to preserve the shape of α¯t for the T = 4000 schedule." 
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        """
        Add noise to images at instant t
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3): 
        """
        This method just implements the diffusion denoising process.
        
        Note that sample() is called with labels = torch.arange(self.num_classes - 1) 
        or in this case [0, 1, 2, 3, 4]. 
        """
        model = self.ema_model if use_ema else self.model
        n = len(labels) 
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device) # Gaussian Noise
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        """
        Take step in model, and EMA model. 
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        """
        Mechanically now we just have the loss is 3x'ed because we run three prompts.
        This is what happens here: 
        - For every image, we have some label, like "male_nurse" or "man"
        - We train for 3 steps, corrsponding to the words this breaks down to. For instance,
        male_nurse = man, nurse, person
        female_nurse = woman, nurse, person
        male_phil = man, philosopher, person
        female_phil = woman, philosopher, person
        man = man, person, man
        woman = woman, person, woman # so only train person on two things
        """
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)

        amt_to_run = 3 if train else 1
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels.to(self.device)

            ## Actually run three steps. Pre-check for None. 
            noneFlag = False
            if np.random.random() < 0.1: # 10% of the time we train with no labels.
                noneFlag = True

            for i in range(amt_to_run): # only do first 2 terms
                with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                    if not noneFlag:
                        labels = torch.tensor([self.step_map[j.item()][i] for j in labels])
                    else:
                        labels = None
                    t = self.sample_timesteps(images.shape[0]).to(self.device)
                    x_t, noise = self.noise_images(images, t)
                    predicted_noise = self.model(x_t, t, labels)
                    loss = self.mse(noise, predicted_noise)
                    avg_loss += loss

                if train:
                    self.train_step(loss)
                    wandb.log({"train_mse": loss.item(),
                                "learning_rate": self.scheduler.get_last_lr()[0]})
                pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        """
        Log images to wandb and save them to disk. Note the num_classes - 1 
        is an artifact of the fact that there are only 5 nouns we care about. 
        """
        labels = torch.arange(self.num_classes - 1).long().to(self.device) # NOTE - this is bespoke
        print(f"Labels: {labels}")
        sampled_images = self.sample(use_ema=False, labels=labels)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        """
        Save model locally and on wandb
        """
        if epoch==-1:
            print("Error")
        else:
            torch.save(self.model.state_dict(), os.path.join("models", f"{run_name}_epoch_{epoch}_ckpt.pt"))
            torch.save(self.ema_model.state_dict(), os.path.join("models", f"{run_name}_epoch_{epoch}_ema_ckpt.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join("models", f"{run_name}_epoch_{epoch}_optim.pt"))
            at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
            at.add_dir(os.path.join("models", run_name))
            wandb.log_artifact(at)

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader) * 3, epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995) # beta = 0.995
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # log predictions
            if epoch % args.log_every_epoch == 0:
                self.log_images()

            # periodic save
            if epoch % args.save_every == 0:
                self.save_model(run_name=args.run_name, epoch=epoch)

        # save model at end
        self.save_model(run_name=args.run_name, epoch=epoch)




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    with wandb.init(project="train_sd", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)
