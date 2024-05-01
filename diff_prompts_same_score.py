import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from enum import Enum
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils_classifier import *
from fairface import *

DataSource = Enum('DataSource', ['DATALOADER', 'RANDOM', 'GENERATIONS'])

class ScoreSimilarity:
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler, 
                 device, num_inference_steps, seed=224):
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

    def eval_similarity_timesteps_dataloader(self, dataloader, prompts, timestep_list, num_data, save_name=None):
        results = {}
        for prompt in prompts:
            for t1 in timestep_list:
                for t2 in timestep_list:
                    results[(prompt, t1, t2)] = []
        with torch.no_grad():
            num_examples = 0
            for data in tqdm(dataloader):
                if num_examples >= num_data:
                    break
                images, _ = data
                images = images[0,:,:,:]
                image_embedding = encode(self.vae, images.unsqueeze(0).to(self.device).float())
                
                diff_timesteps_scores = []
                for prompt in prompts:
                    noise = torch.randn(image_embedding.shape).to(self.device) ## Is this right?
                    for timestep in timestep_list:
                        latents = self.scheduler.add_noise(image_embedding, noise.to(self.device),  
                                                        timesteps=torch.tensor([self.scheduler.timesteps[-timestep]]).to(self.device))
                        latent_model_input = self.scheduler.scale_model_input(latents, timestep=torch.tensor([self.scheduler.timesteps[-timestep]]))
                    
                        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
                                                    truncation=True, return_tensors="pt").to(self.device)
                        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
                        prompt_noise_pred = self.unet(latent_model_input, self.scheduler.timesteps[-timestep], 
                                            encoder_hidden_states=text_embeddings).sample
                        diff_timesteps_scores.append(prompt_noise_pred.detach().clone())
                    for i in range(len(timestep_list)):
                        for j in range(len(timestep_list)):    
                            results[(prompt, timestep_list[i], timestep_list[j])].append(dot_product(diff_timesteps_scores[i], diff_timesteps_scores[j])) 
                num_examples += 1
        if save_name:
            torch.save(results, save_name)
        return results 
    
    def eval_similarity_prompts_dataloader(self, dataloader, prompts, timestep_list, 
                                    num_data, save_name=None):
        results = {}
        for i in range(len(prompts)):
            for j in range(len(prompts)):
                for k in timestep_list:
                    results[(i,j,k)] = []
        with torch.no_grad():
            num_examples = 0
            for data in tqdm(dataloader):
                if num_examples >= num_data:
                    break
                images, _ = data
                images = images[0,:,:,:]
                
                image_embedding = encode(self.vae, images.unsqueeze(0).to(self.device).float())
                for timestep in timestep_list:
                    noise = torch.randn(image_embedding.shape).to(self.device)
                    latents = self.scheduler.add_noise(image_embedding, noise.to(self.device),  
                                                    timesteps=torch.tensor([self.scheduler.timesteps[-timestep]]).to(self.device))
                    latent_model_input = self.scheduler.scale_model_input(latents, timestep=torch.tensor([self.scheduler.timesteps[-timestep]]))
                    prompts_pred = []
                    ## Iterate through prompts
                    for prompt in prompts:
                        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
                                                    truncation=True, return_tensors="pt").to(self.device)
                        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
                        prompt_noise_pred = self.unet(latent_model_input, self.scheduler.timesteps[-timestep], 
                                            encoder_hidden_states=text_embeddings).sample
                        prompts_pred.append(prompt_noise_pred.detach().clone())
                    for i in range(len(prompts)):
                        for j in range(len(prompts)):
                            results[(i,j,timestep)].append(dot_product(prompts_pred[i], prompts_pred[j]))
                num_examples += 1
        
        results["prompts"] = prompts
        if save_name:
            torch.save(results, save_name)
        return results 

# Organizing the data for the heatmaps
def prepare_data_timesteps(data):
    prompts = {}
    for (prompt, t1, t2), values in data.items():
        if prompt not in prompts:
            prompts[prompt] = {}
        if t1 not in prompts[prompt]:
            prompts[prompt][t1] = {}
        if t2 not in prompts[prompt]:
            prompts[prompt][t2] = {}
        prompts[prompt][t1][t2] = np.mean(torch.tensor(values).cpu().numpy())
        prompts[prompt][t2][t1] = np.mean(torch.tensor(values).cpu().numpy())

    return prompts

def prepare_data_prompts(data):
    prompts = {}
    del data["prompts"]
    for (prompt1, prompt2, t), values in data.items():
        if t not in prompts:
            prompts[t] = {}
        if prompt1 not in prompts[t]:
            prompts[t][prompt1] = {}
        if prompt2 not in prompts[t]:
            prompts[t][prompt2] = {}
        prompts[t][prompt1][prompt2] = np.mean(torch.tensor(values).cpu().numpy())
        prompts[t][prompt2][prompt1] = np.mean(torch.tensor(values).cpu().numpy())
    
    return prompts

def plot_multiple_heatmaps(data,grouplabel="",title="",saveas=None):
    # Number of prompts
    num_categories = len(data)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_categories, figsize=(10*num_categories, 8), sharey=True)

    if num_categories == 1:  # If there's only one prompt, axes won't be an array
        axes = [axes]

    # Plotting heatmaps in a loop
    for ax, (prompt, matrix) in zip(axes, data.items()):
        df = pd.DataFrame(matrix).fillna(0)  # Convert to DataFrame and fill NaN values with 0
        sns.heatmap(df, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title(f"Heatmap for {prompt}")
        ax.set_xlabel(f"{grouplabel} 1")
        ax.set_ylabel(f"{grouplabel} 2")

    plt.tight_layout()
    plt.title(title)
    if saveas:
        plt.savefig(saveas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action="store", type=str, required=True, help='Dataset')
    parser.add_argument('--name', action="store", type=str, required=True, help="Label of data")
    parser.add_argument('--timestep_list_file', action="store", type=str, required=True, help='Location of file with timesteps')
    parser.add_argument('--prompts_list_file', action="store", type=str, required=True, help='Location of file with prompts')

    parser.add_argument('--num_data', action="store", type=int, required=False, default=100, help="Number of samples to consider")
    parser.add_argument('--num_timesteps', action="store", type=int, required=False, default=100, help='Number of time steps of diffusion')
    args = parser.parse_args()

    ## Obtain dataloader
    if args.dataset == "FairFace":
        trainloader = obtain_fairface()
    elif args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='/n/holylabs/LABS/sneel_lab/Lab/data', train=True,
                                        download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, num_workers=1)
    
    ## Get filenames 
    similarity_prompts = f"results/{args.name}_sim_diff_prompts_num_data={args.num_data}_num_timesteps={args.num_timesteps}_dataset={args.dataset}"

    ## Open file of timesteps 
    list_of_timesteps = list(map(int, open(args.timestep_list_file).readlines()))

    ## Open file of prompts
    prompts = open(args.prompts_list_file).readlines()
    prompts = list(map(lambda x: x.strip("\n"), prompts))

    ## Loading diffusion model
    print("LOADING DIFFUSION MODEL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, tokenizer, text_encoder, unet, scheduler = load_model("stableV2.1",device)
    score_exp = ScoreSimilarity(vae, tokenizer, text_encoder, unet, scheduler, device, args.num_timesteps)

    ## First experiment
    print("RUNNING PROMPT COMPARISON EXPERIMENTS")
    result_prompts = score_exp.eval_similarity_prompts_dataloader(trainloader, prompts, list_of_timesteps, args.num_data, save_name=similarity_prompts+".pt")
    

    print("CREATING PLOTS")
    prompts_data = prepare_data_prompts(result_prompts)
    plot_multiple_heatmaps(prompts_data,grouplabel="Prompt",saveas=similarity_prompts+".png")


## salloc -p gpu_test -t 0-10:00 --mem 100000 --gres=gpu:1
## python diff_prompts_same_score.py --dataset FairFace --name fairface --timestep_list_file timestep_list_default.txt --prompts_list_file prompts.txt

if __name__== "__main__":
    main()

