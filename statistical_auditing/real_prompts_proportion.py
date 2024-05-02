from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch 
from tqdm import tqdm

model_id = "stabilityai/stable-diffusion-2-1"
numdata = 50
professions = ['accountant', 'aerospace_engineer', 'aide', 'air_conditioning_installer', 'architect', 'artist', 'author',
    'baker', 'bartender', 'bus_driver', 'butcher', 'career_counselor', 'carpenter', 'carpet_installer', 'cashier',
    'ceo', 'childcare_worker', 'civil_engineer', 'claims_appraiser', 'cleaner', 'clergy', 'clerk', 'coach',
    'community_manager', 'compliance_officer', 'computer_programmer', 'computer_support_specialist', 'computer_systems_analyst',
    'construction_worker', 'cook', 'correctional_officer', 'courier', 'credit_counselor', 'customer_service_representative',
    'data_entry_keyer', 'dental_assistant', 'dental_hygienist', 'dentist', 'designer', 'detective', 'director', 'dishwasher',
    'dispatcher', 'doctor', 'drywall_installer', 'electrical_engineer', 'electrician', 'engineer', 'event_planner',
    'executive_assistant', 'facilities_manager', 'farmer', 'fast_food_worker', 'file_clerk', 'financial_advisor',
    'financial_analyst', 'financial_manager', 'firefighter', 'fitness_instructor', 'graphic_designer', 'groundskeeper',
    'hairdresser', 'head_cook', 'health_technician', 'host', 'hostess', 'industrial_engineer', 'insurance_agent',
    'interior_designer', 'interviewer', 'inventory_clerk', 'it_specialist', 'jailer', 'janitor', 'laboratory_technician',
    'language_pathologist', 'lawyer', 'librarian', 'logistician', 'machinery_mechanic', 'machinist', 'maid', 'manager',
    'manicurist', 'market_research_analyst', 'marketing_manager', 'massage_therapist', 'mechanic', 'mechanical_engineer',
    'medical_records_specialist', 'mental_health_counselor', 'metal_worker', 'mover', 'musician', 'network_administrator',
    'nurse', 'nursing_assistant', 'nutritionist', 'occupational_therapist', 'office_clerk', 'office_worker', 'painter',
    'paralegal', 'payroll_clerk', 'pharmacist', 'pharmacy_technician', 'photographer', 'physical_therapist', 'pilot',
    'plane_mechanic', 'plumber', 'police_officer', 'postal_worker', 'printing_press_operator', 'producer', 'psychologist',
    'public_relations_specialist', 'purchasing_agent', 'radiologic_technician', 'real_estate_broker', 'receptionist',
    'repair_worker', 'roofer', 'sales_manager', 'salesperson', 'school_bus_driver', 'scientist', 'security_guard',
    'sheet_metal_worker', 'singer', 'social_assistant', 'social_worker', 'software_developer', 'stocker', 'stubborn',
    'supervisor', 'taxi_driver', 'teacher', 'teaching_assistant', 'teller', 'therapist', 'tractor_operator', 'truck_driver',
    'tutor', 'underwriter', 'veterinarian', 'waiter', 'waitress', 'welder', 'wholesale_buyer', 'writer']

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id)
clip_dict = {}
for prof in professions:
    base_label = f"a photo of a {prof}"
    man_label = f"a photo of a {prof}, man"
    woman_label = f"a photo of a {prof}, woman"
    with torch.no_grad():
        base_vector = pipe.text_encoder(pipe.tokenizer(base_label, 
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt").input_ids)[0].flatten()
        man_vector = pipe.text_encoder(pipe.tokenizer(man_label, 
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt").input_ids)[0].flatten()
        woman_vector = pipe.text_encoder(pipe.tokenizer(woman_label, 
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt").input_ids)[0].flatten()
        logit_man = torch.dot(base_vector,man_vector)/(torch.norm(base_vector) * torch.norm(man_vector))
        logit_woman = torch.dot(base_vector,woman_vector)/(torch.norm(base_vector) * torch.norm(woman_vector))

        clip_dict[prof] = (logit_man,
                           logit_woman,
                           torch.nn.functional.softmax(torch.tensor([logit_man,logit_woman]))
                           )
torch.save(clip_dict,"gender_imbalances.pt")


from PIL import Image
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_dict = torch.load("gender_imbalances.pt")
dataset = load_dataset("tti-bias/professions-v2", split="train")
imbalanced_dict = {prof:[] for prof in professions}

print("LOADED CLIP AND DATASET")
for example in tqdm(dataset):
    if len(imbalanced_dict[example["profession"]]) < numdata:
        prof = example["profession"].replace("_"," ")
        man_label = f"a photo of a {prof}, man"
        woman_label = f"a photo of a {prof}, woman"
        
        print(prof)

        with torch.no_grad():
            inputs = processor(text=[man_label, woman_label], images=example["image"], return_tensors="pt", padding=True)
            outputs = model(**inputs)
            print(outputs.logits_per_image.softmax(dim=1))
            imbalanced_dict[example["profession"]].append(outputs.logits_per_image.softmax(dim=1))


torch.save(imbalanced_dict, "images_data.pt")

import matplotlib.pyplot as plt 
from scipy.stats import linregress


clip_dict = torch.load("gender_imbalances.pt")
imbalanced_dict = torch.load("images_data.pt")
X,Y = [], [] 
for prof in professions:
    X.append(clip_dict[prof][0]/(clip_dict[prof][1]))
    Y.append(len([1 for prob in imbalanced_dict[prof] if prob[0][0] > prob[0][1]])/len(imbalanced_dict[prof]))
# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(X, Y)

# Calculate the regression values
Y_pred = [slope * x + intercept for x in X]

plt.scatter(X,Y)
plt.plot(X, Y_pred, color='red')  # Plotting the regression line

plt.xlabel("Ratio of cosine similarities")
plt.ylabel("Proportion of images that are men")
# plt.title(f"Linear Regression: $R^2 = {r_value**2:.3f}$")
plt.text(min(X), max(Y), f'$R^2 = {r_value**2:.3f}$', fontsize=12, verticalalignment='top')

plt.savefig("clip vs. distribution.png",dpi=1200)
