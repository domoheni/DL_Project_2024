#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J tenflotrial
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u 
### -- send notification at start --
#BSUB -B 
### -- send notification at completion--
#BSUB -N 
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o tenflo_%J.out
#BSUB -e tenflo_%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/11.6

# Activate the virtual environment
source .venv/bin/activate
# Navigate to the project directory
cd add/your/path/

# Run the Python script
python ViT_Train_From_Scrach.py