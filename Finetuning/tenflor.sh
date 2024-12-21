#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100 
### -- set the job Name --
#BSUB -J my_tensorflow_job  
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u marcomalu22@gmail.com
### -- send notification at start --
#BSUB -B marcomalu22@gmail.com
### -- send notification at completion--
#BSUB -N marcomalu22@gmail.com
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o tenflo_%J.out
#BSUB -e tenflo_%J.err
# -- end of LSF options --

module load cuda/12.1
module load cudnn/v8.1.1.33-prod-cuda-11.2


# Create and activate environment
source /dtu/blackhole/01/203777/dl_project/tenflo/tenfloenv/bin/activate

#navigate to the directory
cd /dtu/blackhole/01/203777/dl_project/tenflo

python /dtu/blackhole/01/203777/dl_project/tenflo/resnetFinTun.py