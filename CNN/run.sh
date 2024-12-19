#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100                  # Queue name (modify if needed, e.g., gpua100)
### -- set the job Name --
#BSUB -J deepCNNTrials            # Job name
### -- ask for number of cores (default: 1) --
#BSUB -n 1                        # Number of CPU cores
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU queues
#BSUB -W 24:00
### -- request 10GB of system-memory --
#BSUB -R "rusage[mem=10GB]"
### -- request a 32GB GPU --
#BSUB -R "select[gpu32gb]"
### -- set the output and error file paths. %J is the job ID --
#BSUB -o deepCNN_%J.out
#BSUB -e deepCNN_%J.err

### Activate virtual environment
source /dtu/blackhole/06/203238/dl_project/cnn/.venv/bin/activate

### Navigate to the project directory
cd /dtu/blackhole/06/203238/dl_project/cnn

### Run the Python script
python DeepCNN.py