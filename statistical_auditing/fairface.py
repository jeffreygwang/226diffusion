# Import the load_dataset function from datasets
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from statistical_auditing.utils_classifier import *

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (list): List of all images.
            labels (list): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def download_and_prep():
    # Specify the dataset and config name
    dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 64x64
        transforms.ToTensor(),         # Convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    labels = dataset['train']['gender']
    dataset = CustomImageDataset(dataset['train']['image'], labels, transform=transform)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
    torch.save(trainloader, "gender_fairface.pt")
    print("Dataset creation done!")
    return trainloader

def obtain_fairface():
    # if os.path.exists("fairface.pt"):
    #     print("Loading fairface.pt...")
    #     trainloader = torch.load("fairface.pt")
    # else:
    #     print("Downloading FairFace...")
    trainloader = download_and_prep()
    return trainloader 

if __name__== "__main__":
    trainloader = obtain_fairface()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, tokenizer, text_encoder, unet, scheduler = load_model("stableV2.1",device)

    # race_7 = ["east asian", "indian", "black", "white", "middle eastern", "latino/hispanic", "southeast asian"] 
    gender_2 = ["man", "woman"]
    prompts = [f"photo of a person's face, {race}" for race in gender_2]

    print("RUNNING CLASSIFIER")
    dc = ScoreBasedClassifier(vae, tokenizer, text_encoder, unet, scheduler, device,num_inference_steps=1000)

    tsteps = [5]
    arr = []
    scores = []


    for t in tsteps:
        score_arr, acc = dc.evaluate_performance(trainloader, prompts, uncond_prompt="photo of a person's face", method="convex_opt", timestep=t, num_data=250)
        arr.append(acc)
        scores.append(score_arr)
        print(acc)
    