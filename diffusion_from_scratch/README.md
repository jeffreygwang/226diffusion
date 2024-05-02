See main README for details.

To train, just run `python ddpm_conditional.py`.

The main difference between this implementation of a diffusion model and an off-the-shelf CIFAR-10 one is that we use pre-made embeddings. In particular, the base implementation takes as input to the UNet (the neural network that learns the score) some vector $v \in \mathbb{R}^{256}$ where $v = f_{\text{pos}}(t) + f_{\text{embed}}(x)$, where $f_{\text{pos}}: \mathbb{R} \to \mathbb{R}^{256}$ is a sinusoidal positional encoding of the time and $f_{\text{embed}}: \mathbb{R} \to \mathbb{R}^{256}$ is an embedding of class (a number from 0 to 9) in 256-dimensional space that gets learned as the model runs. In our diffusion model, we changed the positional encoding of time to instead be of the form $f_{\text{pos}}: \mathbb{R} \to \mathbb{R}^{128}$, and concatenate the positional encoding with a text embedding of the input prompt in $\mathbb{R}^{128}$. Since every embedding in w2vNEWS is in $\mathbb{R}^{300}$, we use a Johnson-Lindenstrauss projection to reduce its dimension to 128. 

To make the model train faster and better, we used standard hyperparameters found in Nichol et al 2021 and elsewhere. 
- Batch size of 50. 
- Max learning rate of $1e-4$ on a 1cycle learning rate scheduler 
- Keep an exponential moving average of models for stability, with $\beta = 0.995$. 

