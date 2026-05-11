#!/bin/bash
cd /home/user/noob/deraining/models/NAFNet
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
nohup torchrun --nproc_per_node=1 --master_port=29501 \
    basicsr/train.py -opt ../../configs/nafnet_derain_w32.yml --launcher pytorch \
    > /tmp/nafnet_train.log 2>&1 &
echo "NAFNet training PID: $!"
