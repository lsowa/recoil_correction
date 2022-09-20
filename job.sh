#!/bin/bash

xrdcp -rfv root://ceph-node-a.etp.kit.edu://lsowa/recoil/mc.root .
xrdcp -rfv root://ceph-node-a.etp.kit.edu://lsowa/recoil/dt.root .

#python3 -m venv environment
#source environment/bin/activate
/cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc8-opt/bin/python3.9 -m venv enviroment
. enviroment/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

batch = 50000
flows = 8
nn_hidden = 3
nn_nodes = 200

test = 2

python recoil.py --batch $batch --flows $flows --nn-hidden $nn_hidden --nn-nodes $nn_nodes --test $test
python evaluation.py --flows $flows --nn-hidden $nn_hidden --nn-nodes $nn_nodes --test $test --model 'output/model.pt' --output 'output/'

