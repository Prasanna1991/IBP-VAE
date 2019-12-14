# IBP-VAE
Improving Disentangled Representatoin Learning with the Beta Bernoulli Process [[link]](https://arxiv.org/abs/1909.01839).

The most of the contents in this code base are copied from this [repository](https://github.com/rachtsingh/ibp_vae), so credit goes to them! 

# Set the conda environment 
Use the environment.yml file provided in this repo to set the conda environment to run the code. 

`conda env create -f environment.yml`

Finally, to run the code (e.g., IBP-VAE for MNIST with beta = 5):

`python IBP-VAE-MNIST.py --beta 5`

# Compilation
The most important part in this code is the compilation in order to run the code with GPU support. For this you need to navigate to `lgamma` and run `./make.sh`. You might need to change architecture setting inside `make.sh` according to your GPU card's compute capability and the `CUDA_PATH` might also need to be customized. Please refer to the original code base for IBP-VAE [[link]](https://github.com/rachtsingh/ibp_vae) if you have any troble compiling or contact me at [pkg2182@rit.edu](pkg2182@rit.edu). 
