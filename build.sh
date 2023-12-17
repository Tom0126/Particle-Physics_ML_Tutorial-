#!/bin/bash

source /lustre/collider/songsiyuan/conda.env

# Specify the name of the virtual environment
environment_name="env_test"

# Create a virtual environment
conda create --name $environment_name python=3.8

# Activate the virtual environment
conda activate $environment_name

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# List of packages to install using conda
conda_packages=("numpy" "pandas" "matplotlib" "scikit-learn")

# List of packages to install using pip
pip_packages=("torchsummary" "torchmetrics" "xgboost")

# Install specified packages using conda
conda install "${conda_packages[@]}"

# Install specified packages using pip
pip install "${pip_packages[@]}"



# Display information
echo "Virtual environment '$environment_name' has been created, and packages including PyTorch have been installed."
echo "Activate the environment using 'conda activate $environment_name'."
