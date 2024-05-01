import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
from enum import Enum
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random 

from utils_classifier import *
from fairface import *


class DiffusionELBOClassifier:
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler, 
                 device, num_inference_steps,seed=224):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        if self.scheduler:
            self.scheduler.set_timesteps(num_inference_steps)
        self.device = device

        self.pipeline = StableDiffusionPipeline(self.vae,self.text_encoder,self.tokenizer,self.unet,self.scheduler,None,None)
        
        torch.manual_seed(seed)

    def evaluate_performance(self, dataloader, prompts, n_samples=100, num_data=500,loss="l2"):
        logit_arrs = []

        # Expects batch size of 1
        with torch.no_grad():
            num_examples = 0
            num_correct = 0
            for data in tqdm(dataloader):
                if num_examples >= num_data:
                    break
                images, labels = data
                images = images[0,:,:,:]
                true_class = labels[0]
                
                ## Generate image embedding
                image_embedding = encode(self.vae, images.unsqueeze(0).to(self.device).float())
                losses = []
                for _ in range(n_samples):
                    timestep = random.randint(1,self.num_inference_steps)
                    scheduler_timesteps = torch.tensor([self.scheduler.timesteps[timestep]]).to(self.device)
                    
                    ## Generate noise
                    noise = torch.randn(image_embedding.shape).to(self.device) 
                    latents = self.scheduler.add_noise(image_embedding, noise.to(self.device),  
                                                    timesteps=scheduler_timesteps)
                    latent_model_input = self.scheduler.scale_model_input(latents, timestep=scheduler_timesteps)
                    latent_model_input = latent_model_input.repeat([len(prompts),1,1,1])

                    ## Compute unconditional noise value
                    prompts_tokens = self.tokenizer(prompts, padding="max_length", 
                                                    max_length=self.tokenizer.model_max_length, return_tensors="pt")
                    prompts_embeds = self.text_encoder(prompts_tokens.input_ids.to(self.device))[0]
                    noise_preds = self.unet(latent_model_input, scheduler_timesteps, 
                                            encoder_hidden_states=prompts_embeds).sample
                    
                    if loss == 'l2':
                        error = -F.mse_loss(noise.repeat((len(prompts),1,1,1)), noise_preds, reduction='none').mean(dim=(1, 2, 3))
                    elif loss == 'l1':
                        error = -F.l1_loss(noise.repeat((len(prompts),1,1,1)), noise_preds, reduction='none').mean(dim=(1, 2, 3))
                    elif loss == 'huber':
                        error = -F.huber_loss(noise.repeat((len(prompts),1,1,1)), noise_preds, reduction='none').mean(dim=(1, 2, 3))
                    losses.append(error.detach().cpu())
                losses = torch.stack(losses).T
                loss_means = losses.mean(dim=1)
                
                ## Compute probs
                probs = torch.nn.functional.softmax(loss_means,dim=-1)

                num_examples += 1
                logit_arrs.append(loss_means)
                print(f"Logits: {loss_means}")
                print(f"Probs: {probs}")
                print(f"Argmax: {torch.tensor(probs).argmax()}")
                print(f"True Class: {true_class}")
                if torch.tensor(probs).argmax()==true_class:
                    num_correct += 1
                    print("CORRECT")
            print(f"Step {timestep} Accuracy : {num_correct/num_examples}")
        return logit_arrs, num_correct/num_examples

                
if __name__== "__main__":
    trainset = torchvision.datasets.CIFAR10(root='/n/holylabs/LABS/sneel_lab/Lab/data', train=False,
                                            download=True, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, num_workers=1)

    ## Loading components
    print("LOADING COMPONENTS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, tokenizer, text_encoder, unet, scheduler = load_model("stableV2.1",device,euler=True)
    
    dc = DiffusionELBOClassifier(vae, tokenizer, text_encoder, unet, scheduler, device,num_inference_steps=1000)
    prompts = ["a blurry photo of a airplane.", "a blurry photo of a automobile.", \
               "a blurry photo of a bird.", "a blurry photo of a cat.", 
               "a blurry photo of a deer.", "a blurry photo of a dog.", \
                "a blurry photo of a frog.", "a blurry photo of a horse.",\
                "a blurry photo of a ship.", "a blurry photo of a truck."]
    dc.evaluate_performance(trainloader, prompts, loss="l2")
    # dc.evaluate_performance(trainloader, ["airplane", "automobile", "bird", 
    #                                     "cat", "deer", "dog", 
    #                                     "frog", "horse", "ship", "truck"],loss="l1")
    # dc.evaluate_performance(trainloader, ["airplane", "automobile", "bird", 
    #                                     "cat", "deer", "dog", 
    #                                     "frog", "horse", "ship", "truck"],loss="huber")



## salloc -p gpu_test -t 0-10:00 --mem 100000 --gres=gpu:1