# LFP
This code is an official implementation of  "**LFP: LOOP FEATURE PYRAMID FOR OBJECT DETECTION** (Submission under review [ICIP2021](https://2021.ieeeicip.org/))" based on the open source object detection toolbox mmdetection.

![image](https://github.com/huitang96/LFP/blob/master/LFP/MY_PICTURES/images/3_3.bmp)    ![image](https://github.com/huitang96/LFP/blob/master/LFP/MY_PICTURES/images/8_8.bmp)   ![image](https://github.com/huitang96/LFP/blob/master/LFP/MY_PICTURES/images/5_5.bmp)    ![image](https://github.com/huitang96/LFP/blob/master/LFP/MY_PICTURES/images/3.bmp)       ![image](https://github.com/huitang96/LFP/blob/master/LFP/MY_PICTURES/images/8.bmp)         ![image](https://github.com/huitang96/LFP/blob/master/LFP/MY_PICTURES/images/5.bmp)

**Notes:  PAFPN(top), LFP(bottom)**


# Installation
Please refer to[INSTALL.md ](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)of mmdetection.

I use pytorch1.5.0, cuda10.2, and mmcv1.2.2.


# Train and Inference
Please use the following commands for training and testing by single GPU or multiple GPUs.

## Train with a single GPU
     python tools/train.py ${CONFIG_FILE}  
     
## Train with multiple GPUs
     ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
     
## Test with a single GPU
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]   
    
## Test with multiple GPUs
    ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] 
   
