import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler
from PIL import Image
import os
from enum import Enum
import psutil
import cvxpy as cp
import numpy as np


def mem_stats():
    '''
    Memory statistics for memory management
    '''
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    r = torch.cuda.memory_reserved(0) / 1024**3
    a = torch.cuda.memory_allocated(0) / 1024**3
    print(f"CPU RAM: {psutil.virtual_memory()[3]/1e9}/{psutil.virtual_memory()[4]/1e9:.2f} ({psutil.virtual_memory()[2]:.2f}%)\n"
          f"Total Memory: {t:.2f} GB\n"
          f"Reserved Memory: {r:.2f} GB ({(100*(r/t)):.2f}%)\n"
          f"Remaining Memory: {t-r:.2f} GB ({(100*(t-r)/t):.2f}%)\n"
          f"---------------------------------\n"
          f"Allocated Memory: {a:.2f} GB ({(100*(a/t)):.2f}%)\n"
          f"Percent of Reserved Allocated: {(100*(a+1e-9)/(r+1e-9)):.2f}%\n")

def pil_to_tensor(image):
    """
    Convert a PIL Image to a PyTorch Tensor without using torchvision.
    """
    # Convert the PIL image to a NumPy array
    numpy_image = np.array(image)

    # Convert the NumPy array to a PyTorch tensor
    tensor_image = torch.tensor(numpy_image)

    # Convert the tensor to float and scale it to [0, 1]
    tensor_image = tensor_image.float() / 255.0

    # Change the layout from (H x W x C) to (C x H x W)
    if tensor_image.ndimension() == 3:
        tensor_image = tensor_image.permute(2, 0, 1)

    return tensor_image

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model_type,device,total_time=1000,ddim=False,euler=False):
    if model_type == "stableV2.1":
        link = "stabilityai/stable-diffusion-2-1"
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1",subfolder="vae").to(device)
    elif model_type == "stableXL":
        assert False 
        link = "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type=="animagine":
        link = "cagliostrolab/animagine-xl-3.0"
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(link,subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(link,subfolder="text_encoder").to(device)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet").to(device)

    # 4. The Scheduler for generating the images.
    if ddim:
        print("USING DDIM")
        ddim_scheduler_config = {
                "_class_name": "DDIMScheduler",
                "_diffusers_version": "0.8.0",
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "prediction_type": "v_prediction",
                "set_alpha_to_one": False,
                "skip_prk_steps": True,
                "steps_offset": 1,
                "trained_betas": None
                }
        scheduler = DDIMScheduler.from_config(ddim_scheduler_config, rescale_betas_zero_snr=True, timestep_spacing="trailing")        
    elif euler:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_type, subfolder="scheduler")
    else:
        scheduler = DDPMScheduler.from_pretrained(link, subfolder="scheduler")
    scheduler.set_timesteps(total_time)
    
    return vae, tokenizer, text_encoder, unet, scheduler

def encode(vae,input_images):
    '''Encodes images into latent space'''
    if input_images.shape[1]==1:
        input_images = input_images.repeat(1,3,1,1)
    with torch.no_grad():
        latent = vae.encode(input_images*2 - 1) # Note scaling
    return vae.config.scaling_factor * latent.latent_dist.sample()

def decode(vae,latents):
    '''Decodes latent space into images'''
    latents = (1 / vae.config.scaling_factor) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image

def dot_product(uncond, cond):
    return torch.dot(uncond.flatten(), cond.flatten())/(torch.norm(uncond)*torch.norm(cond)).detach().clone().cpu()

def norm_diff(uncond, cond,p=2):
    return -torch.norm(uncond-cond,p=p)

def func_over_prompts(uncond, conds, f):
    return torch.tensor([f(uncond,cond) for cond in conds])


def solve(X,y):
    # Number of variables in x
    n = X.shape[1]

    # Define the optimization variable
    x = cp.Variable(n)

    # Define the objective function (minimize the L2 norm of Ax-b)
    objective = cp.Minimize(cp.norm(X @ x - y, 2))

    # Define the constraints
    constraints = [cp.sum(x) == 1, x >= 0]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()
    
    # Print the optimal value of x
    return torch.tensor(x.value)

def pca_linear_program(uncond, cond,dim=2,normalize=True):
    all_vectors = torch.stack([uncond]+cond).flatten(start_dim=1)
    means = all_vectors.mean(axis=0)
    all_vectors -= means[None,:].repeat((len(cond)+1,1))
    if normalize:
        all_vectors = torch.nn.functional.normalize(all_vectors)
    
    ## Apply pca and project
    U,S,V = torch.pca_lowrank(all_vectors)
    projected_vectors = all_vectors @ V[:,:dim]
    return linear_program( projected_vectors[0,:], [projected_vectors[i,:] for i in range(1,len(cond)+1)])

def linear_program(uncond, conds):
    X = torch.stack(conds).flatten(start_dim=1).T.cpu().numpy()
    y = uncond.flatten().cpu().numpy()
    
    return solve(X,y)
    

class ScoreBasedClassifier:
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

    # def similarity_over_t(self, dataloader, prompts, timesteps, num_data=500):
    #     similarity_arrs = {t: [] for t in timesteps}

    #     # Expects batch size of 1
    #     with torch.no_grad():
    #         num_examples = 0
    #         for data in tqdm(dataloader):
    #             if num_examples >= num_data:
    #                 break
    #             images, labels = data
    #             images = images[0,:,:,:]
    #             prompts_pred = []

    #             ## Generate image embedding
    #             image_embedding = encode(self.vae, images.unsqueeze(0).to(self.device).float())
    #             noise = torch.zeros(image_embedding.shape).to(self.device)
                
    #             latents = self.scheduler.add_noise(image_embedding, noise.to(self.device),  
    #                                                timesteps=torch.tensor([self.scheduler.timesteps[-timestep]]).to(self.device))
    #             latent_model_input = self.scheduler.scale_model_input(latents, timestep=torch.tensor([self.scheduler.timesteps[-timestep]]))

    #             ## Iterate through prompts
    #             for prompt in prompts:
    #                 text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
    #                                             truncation=True, return_tensors="pt").to(self.device)
    #                 text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
    #                 prompt_noise_pred = self.unet(latent_model_input, self.scheduler.timesteps[-timestep], 
    #                                        encoder_hidden_states=text_embeddings).sample
    #                 prompts_pred.append(prompt_noise_pred.detach().clone())



    #             num_examples += 1
    #     return similarity_arrs
                
    def evaluate_performance(self, dataloader, prompts, uncond_prompt="", method="convex_opt",timestep=20, num_data=500):
        scores_arrs = []

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
                noise = torch.randn(image_embedding.shape).to(self.device) ## should we add random noise or not???
                
                latents = self.scheduler.add_noise(image_embedding, noise.to(self.device),  
                                                   timesteps=torch.tensor([self.scheduler.timesteps[-timestep]]).to(self.device))
                latent_model_input = self.scheduler.scale_model_input(latents, timestep=torch.tensor([self.scheduler.timesteps[-timestep]]))

                ## Compute unconditional noise value
                uncond_input = self.tokenizer([uncond_prompt], padding="max_length", 
                                              max_length=self.tokenizer.model_max_length, return_tensors="pt")
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                uncond_noise_pred = self.unet(latent_model_input, torch.tensor([self.scheduler.timesteps[-timestep]]).to(self.device), 
                                              encoder_hidden_states=uncond_embeddings).sample
                

                prompts_pred = []
                ## Iterate through prompts
                for prompt in prompts:
                    text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
                                                truncation=True, return_tensors="pt").to(self.device)
                    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
                    prompt_noise_pred = self.unet(latent_model_input, self.scheduler.timesteps[-timestep], 
                                           encoder_hidden_states=text_embeddings).sample
                    prompts_pred.append(prompt_noise_pred.detach().clone())
                    
                if method == "dot product":
                    prompt_scores = func_over_prompts(uncond_noise_pred, prompts_pred, dot_product)
                elif method == "pca":
                    prompt_scores = pca_linear_program(uncond_noise_pred, prompts_pred)
                elif type(method)==int:
                    prompt_scores = func_over_prompts(uncond_noise_pred, prompts_pred, lambda a,b: norm_diff(a, b,p=method))
                elif method == "convex_opt":
                    prompt_scores = linear_program(uncond_noise_pred, prompts_pred)
                else:
                    assert False 
                
                num_examples += 1
                scores_arrs.append(prompt_scores)
                print(f"Scores: {prompt_scores}")
                print(f"Argmax: {torch.tensor(prompt_scores).argmax()}")
                print(f"True Class: {true_class}")
                if torch.tensor(prompt_scores).argmax()==true_class:
                    num_correct += 1
                    print("CORRECT")
            print(f"Method {method} Step {timestep} Accuracy : {num_correct/num_examples}")
        return scores_arrs, num_correct/num_examples

                




