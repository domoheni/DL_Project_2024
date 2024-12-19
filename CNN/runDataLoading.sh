#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100                  # Queue name (change if needed)
### -- set the job Name --
#BSUB -J datasetRestructure       # Job name
### -- ask for number of cores (default: 1) --
#BSUB -n 1                        # Number of CPU cores
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU queues
#BSUB -W 1:00
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=5GB]"
### -- set the output and error file paths. %J is the job ID --
#BSUB -o datasetRestructure_%J.out
#BSUB -e datasetRestructure_%J.err

### Activate virtual environment
source /dtu/blackhole/06/203238/dl_project/cnn/.venv/bin/activate

### Navigate to the project directory
cd /dtu/blackhole/06/203238/dl_project/cnn

### Run the Python script
python data_loading.py
