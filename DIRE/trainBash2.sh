#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_faces
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u domoheni@gmail.com
### -- send notification at start --
#BSUB -B domoheni@gmail.com
### -- send notification at completion--
#BSUB -N domoheni@gmail.com
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_faces%J.out
#BSUB -e train_faces%J.err
# -- end of LSF options --

# Load the CUDA module
module load cuda/11.6

#module load cuda/11.6
source .venv/bin/activate

EXP_NAME="custom_experiment"
DATASETS="/dtu/blackhole/10/203248/dl_project/DIRE/test_data2/train"
DATASETS_TEST="/dtu/blackhole/10/203248/dl_project/DIRE/test_data2/test"

python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST