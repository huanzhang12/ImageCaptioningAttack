
This is our code for image captioning attack. Our paper can be found on arxiv.

First please clone this directory.

To run the attack on MSCOCO dataset, you need to bash run_attack.sh. Before you do this, please go to run_attack.sh and specify 4 paths:

1. ${VOCAB_FILE} is the path to vocabulary file generated by the preprocessing script. (it should be a word_counts.txt)

2. ${CHECKPOINT_PATH} is the path to the checkpoint file.

3. ${CAPTION_FILE} is the path to the validation set's caption file (for example ../coco-caption/annotations/captions_val2014.json)

4. ${IMAGE_DIRECTORY} is the directory of MSCOCO validation set (for example ../mscoco/image/val2014/)

In this code we provide 4 attack modes: targeted caption attack, untargeted caption attack, targeted keyword attack and untargeted keyword attack. 

We have two boolean parameters to control the attack mode, "use_keywords" and "targeted". For example, "--use_keywords=False" and "--targeted=True" give you targeted caption attack. 

We also have a parameter "use_logits" for you to choose between the logits loss or log-prob loss. To use logits loss, simply add "--use_logits=True" and to use log-prob loss, add "--use_logits=False". The detailed form of our losses can be find in our paper.

"result_directory" is the directory we save the results. We also add a /fail_log directory in the result directory to save the log to failed attacks. 

When we do the experiments on MSCOCO validation set, we first randomly shuffle the images. Then we pick images in this queue one by one to attack. "exp_num" detemines the number of experiments. One experiment means we attack one image.  