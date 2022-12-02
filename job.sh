#!/bin/bash

# argumentis:
#  0 
#  1 number of available GPUs

# set variables
HOME=$PWD #set home dir
CUDA_VISIBLE_DEVICES=0 #set first cuda device
end=$(expr $1 - 1) # add more devices if available
for i in $(seq 1 1 $end); do CUDA_VISIBLE_DEVICES=($CUDA_VISIBLE_DEVICES,$i); done


xrdcp -rfv root://ceph-node-a.etp.kit.edu://lsowa/recoil/mc.root .
xrdcp -rfv root://ceph-node-a.etp.kit.edu://lsowa/recoil/dt.root .

#python3 -m venv environment
#source environment/bin/activate
/cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc8-opt/bin/python3.9 -m venv enviroment
. enviroment/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# wandb setup
wandb login 4b3d06ec7ab93fa2a8a3a11adee369183d4f3247


batch=50000
flows=8
nn_hidden=3
nn_nodes=200
testrun=0

python recoil.py --batch $batch --flows $flows --nn-hidden $nn_hidden --nn-nodes $nn_nodes --test $testrun --ndevices $1
python evaluation.py --flows $flows --nn-hidden $nn_hidden --nn-nodes $nn_nodes --test $testrun --model 'output/model.pt' --output 'output/'

