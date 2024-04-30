# diffusion_model_from_scratch

# Training a Diffusion Model from scratch

Goal:
- We know that words with biased embeddings (e.g. nurse) tend to lead to generations that are biased (e.g. more women).
- However, this could be because the underlying diffusion training data is biased. 
- What if we trained a diffusion model with balanced training data? And then trained it conditional on word embeddings for the given images. 
- We evaluate on philosopher (bias male), person (balanced), nurse (biased female)

# Synthetic data generation

Diffusion models are data hungry. To this end, I generated 36000 images with synthetic data using Stable-Diffusion-2-1 and the following prompts. There were 6000 of each type: male nurse, female nurse, male philosopher, female philosopher, . 

```
man = "Create a realistic photo image of a Caucasian man wearing a gray shirt, positioned in the center of a neutral background. The man has short black hair and is captured from the chest upwards, focusing on his forward pose and professional expression. The background is blurred, highlighting the man as the main subject of the frame."
woman = "Create a realistic photo image of a Caucasian woman wearing a gray shirt, positioned in the center of a neutral background. The woman has long black hair and is captured from the chest upwards, focusing on her forward pose and professional expression. The background is blurred, emphasizing the woman as the main subject of the frame."
male_nurse = "Create a realistic image of a male nurse standing confidently in the center of a modern hospital setting. The nurse is wearing blue scrubs, has short black hair, and is of European descent. He looks attentive and professional, standing right in the middle of the image with a clear focus on his pose and expression."
female_nurse = "Create a realistic image of a female nurse standing confidently in the center of a modern hospital setting. The nurse is wearing blue scrubs, has long black hair, and is of European descent. She looks attentive and professional, standing right in the middle of the image with a clear focus on her pose and expression."
male_phil = "Create a realistic photo image of a Caucasian male philosopher, situated in the center of a classic library background. He wears a tweed jacket and has short black hair, neatly styled. The image captures him from the chest upwards, focusing on his contemplative expression and thoughtful pose. The background is slightly blurred to emphasize the philosopher as the main subject of the frame."
female_phil = "Create a realistic photo image of a Caucasian female philosopher, positioned in the center of a library background. She wears a tweed jacket and has long black hair, neatly styled. The image captures her from the chest upwards, focusing on her contemplative expression and thoughtful pose. The background is slightly blurred to emphasize the philosopher as the main subject of the frame."
```

# Mechanics of Training

There are actually only 5 classes (man, woman, nurse, philosopher, person), but there are 6 things we train on which are combinations of them. Hence because I am conditioning on WORD embeddings, I train each image on 3 separate steps with this breakdown: 

```
male_nurse = man, nurse
female_nurse = woman, nurse
male_phil = man, philosopher
female_phil = woman, philosopher
man = man, person
woman = woman, person
```

Then the actual forward pass runs like this: 
- Before anything, I first take the w2vec-google-news-300 embeddings of these guys, and JL project the 5 words to 128 space. 
- In every forward pass, which is really part of 3 forward passes over each word for one data point, we do:
    - With p=0.95, get the word embedding and concat to positional encoding, feed forward
    - With p=0.05, just take pos encoding + all 0's, feed forward

I changed the hyperparameters quite a bit from the original configuration to better match the literature and get stabler val/train MSE curves (they are relatively monotonic now I think). Also, there's an EMA model for stability being trained too. 

