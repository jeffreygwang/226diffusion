from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverSDEScheduler
import cv2
import matplotlib.pyplot as plt
import os
import torch

model_id = "stabilityai/stable-diffusion-2-1"

# Use the Euler scheduler here instead
scheduler = DPMSolverSDEScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generations
N = 400

# Prompts
man = "Create a realistic photo image of a Caucasian man wearing a gray shirt, positioned in the center of a neutral background. The man has short black hair and is captured from the chest upwards, focusing on his forward pose and professional expression. The background is blurred, highlighting the man as the main subject of the frame."
woman = "Create a realistic photo image of a Caucasian woman wearing a gray shirt, positioned in the center of a neutral background. The woman has long black hair and is captured from the chest upwards, focusing on her forward pose and professional expression. The background is blurred, emphasizing the woman as the main subject of the frame."
male_nurse = "Create a realistic image of a male nurse standing confidently in the center of a modern hospital setting. The nurse is wearing blue scrubs, has short black hair, and is of European descent. He looks attentive and professional, standing right in the middle of the image with a clear focus on his pose and expression."
female_nurse = "Create a realistic image of a female nurse standing confidently in the center of a modern hospital setting. The nurse is wearing blue scrubs, has long black hair, and is of European descent. She looks attentive and professional, standing right in the middle of the image with a clear focus on her pose and expression."
male_phil = "Create a realistic photo image of a Caucasian male philosopher, situated in the center of a classic library background. He wears a tweed jacket and has short black hair, neatly styled. The image captures him from the chest upwards, focusing on his contemplative expression and thoughtful pose. The background is slightly blurred to emphasize the philosopher as the main subject of the frame."
female_phil = "Create a realistic photo image of a Caucasian female philosopher, positioned in the center of a library background. She wears a tweed jacket and has long black hair, neatly styled. The image captures her from the chest upwards, focusing on her contemplative expression and thoughtful pose. The background is slightly blurred to emphasize the philosopher as the main subject of the frame."

bs = 15

# Results Directory
directory = "Images"
if not os.path.exists(directory):
    os.makedirs(directory)

def get_inputs(batch_size, prompt):                                                                                                                                                                                                                 
  generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
  prompts = batch_size * [prompt]                                                                                                                                                                                                             
  num_inference_steps = 20                                                                                                                                                                                                                  

  return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}   

giant_save_file = []
total_generation = 0
for i in range(N * 6):
    if i % 6 == 0:
        imgid = i // 6
        images = pipe(**get_inputs(bs, man)).images
        for image in images:
            total_generation += 1
            starter = f"{directory}/man_{imgid}_{total_generation}"
            file = f"{starter}.png"
            image.save(file)

            # re-scale and save
            img = cv2.imread(file)
            img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            # rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(f"{starter}_64x64.png", rgb_image)
            cv2.imwrite(f"{starter}_64x64.png", img64)

            # 128 x 128
            img = cv2.imread(file)
            img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            # rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_128x128.png", img128)
    elif i % 6 == 1:
        imgid = i // 6
        images = pipe(**get_inputs(bs, male_nurse)).images
        for image in images:
            total_generation += 1
            starter = f"{directory}/male_nurse_{imgid}_{total_generation}"
            file = f"{starter}.png"
            image.save(file)

            # re-scale and save
            img = cv2.imread(file)
            img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_64x64.png", img64)

            # 128 x 128
            img = cv2.imread(file)
            img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_128x128.png", img128)
    elif i % 6 == 2:
        imgid = i // 4
        images = pipe(**get_inputs(bs, woman)).images
        for image in images:
            total_generation += 1
            starter = f"{directory}/woman_{imgid}_{total_generation}"
            file = f"{starter}.png"
            image.save(file)

            # re-scale and save
            img = cv2.imread(file)
            img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_64x64.png", img64)

            # 128 x 128
            img = cv2.imread(file)
            img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_128x128.png", img128)
    elif i % 6 == 3:
        imgid = i // 6
        images = pipe(**get_inputs(bs, female_nurse)).images
        for image in images:
            total_generation += 1
            starter = f"{directory}/female_nurse_{imgid}_{total_generation}"
            file = f"{starter}.png"
            image.save(file)

            # re-scale and save
            img = cv2.imread(file)
            img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_64x64.png", img64)

            # 128 x 128
            img = cv2.imread(file)
            img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_128x128.png", img128)
    elif i % 6 == 4:
        imgid = i // 6
        images = pipe(**get_inputs(bs, male_phil)).images
        for image in images:
            total_generation += 1
            starter = f"{directory}/male_phil_{imgid}_{total_generation}"
            file = f"{starter}.png"
            image.save(file)

            # re-scale and save
            img = cv2.imread(file)
            img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_64x64.png", img64)

            # 128 x 128
            img = cv2.imread(file)
            img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_128x128.png", img128)
    else:
        imgid = i // 6
        images = pipe(**get_inputs(bs, female_phil)).images
        for image in images:
            total_generation += 1
            starter = f"{directory}/female_phil_{imgid}_{total_generation}"
            file = f"{starter}.png"
            image.save(file)

            # re-scale and save
            img = cv2.imread(file)
            img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_64x64.png", img64)

            # 128 x 128
            img = cv2.imread(file)
            img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            #rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{starter}_128x128.png", img128)




# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverSDEScheduler
# import cv2
# import matplotlib.pyplot as plt

# model_id = "stabilityai/stable-diffusion-2-1"

# # Use the Euler scheduler here instead
# scheduler = DPMSolverSDEScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# # Generations
# N = 5000

# # Prompts
# man = "Create a realistic photo image of a Caucasian man wearing a gray shirt, positioned in the center of a neutral background. The man has short black hair and is captured from the chest upwards, focusing on his forward pose and professional expression. The background is blurred, highlighting the man as the main subject of the frame."
# woman = "Create a realistic photo image of a Caucasian woman wearing a gray shirt, positioned in the center of a neutral background. The woman has long black hair tied back in a ponytail and is captured from the chest upwards, focusing on her forward pose and professional expression. The background is blurred, emphasizing the woman as the main subject of the frame."
# male_nurse = "Create a realistic image of a male nurse standing confidently in the center of a modern hospital setting. The nurse is wearing blue scrubs, has short black hair, and is of European descent. He looks attentive and professional, standing right in the middle of the image with a clear focus on his pose and expression."
# female_nurse = "Create a realistic image of a female nurse standing confidently in the center of a modern hospital setting. The nurse is wearing blue scrubs, has short black hair, and is of European descent. She looks attentive and professional, standing right in the middle of the image with a clear focus on his pose and expression."

# # Results Directory
# directory = "Images"

# def get_inputs(batch_size=1, prompt):                                                                                                                                                                                                                 
#   generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
#   prompts = batch_size * [prompt]                                                                                                                                                                                                             
#   num_inference_steps = 50                                                                                                                                                                                                                    

#   return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}   

# giant_save_file = []

# for i in range(N * 4):
#     if i % 4 == 0:
#         imgid = i // 4
#         image = pipe(man).images[0]
#         starter = f"{directory}/man_{imgid}"
#         file = f"{starter}.png"
#         image.save(file)

#         # re-scale and save
#         img = cv2.imread(file)
#         img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_64x64.png", rgb_image)

#         # 128 x 128
#         img = cv2.imread(file)
#         img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_128x128.png", rgb_image)
#     elif i % 4 == 1:
#         imgid = i // 4
#         image = pipe(male_nurse).images[0]
#         starter = f"{directory}/male_nurse_{imgid}"
#         file = f"{starter}.png"
#         image.save(file)

#         # re-scale and save
#         img = cv2.imread(file)
#         img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_64x64.png", rgb_image)

#         # 128 x 128
#         img = cv2.imread(file)
#         img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_128x128.png", rgb_image)
#     elif i % 4 == 2:
#         imgid = i // 4
#         image = pipe(woman).images[0]
#         starter = f"{directory}/woman_{imgid}"
#         file = f"{starter}.png"
#         image.save(file)

#         # re-scale and save
#         img = cv2.imread(file)
#         img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_64x64.png", rgb_image)

#         # 128 x 128
#         img = cv2.imread(file)
#         img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_128x128.png", rgb_image)
#     else:
#         imgid = i // 4
#         image = pipe(female_nurse).images[0]
#         starter = f"{directory}/female_nurse_{imgid}"
#         file = f"{starter}.png"
#         image.save(file)

#         # re-scale and save
#         img = cv2.imread(file)
#         img64 = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_64x64.png", rgb_image)

#         # 128 x 128
#         img = cv2.imread(file)
#         img128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
#         rgb_image = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{starter}_128x128.png", rgb_image)



