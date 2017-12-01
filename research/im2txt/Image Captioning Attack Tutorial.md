
This is our code for image captioning attack. Our paper can be found on arxiv.
To run the attack, simply bash run_attack.sh
In this code we provide 4 attack modes: targeted caption attack, untargeted caption attack, targeted keyword attack and untargeted keyword attack. 
We have two boolean parameters to control the attack mode, "use_keywords" and "targeted". For example, "--use_keywords=False" and "--targeted=True" give you targeted caption attack. 
