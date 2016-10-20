import os
runs = [("nin_imagenet", "NiN"), ("alexnet_ng_conv","AlexNet_no_groups"),("googlenet_conv","GoogLeNet"),]#("vgg_19","VGG-19")]
imgs = [1,5,20]

os.system( "rm op_sigs_full.txt" );
for model_dir,net_name in runs:
    for img in imgs:
        os.system( "boda run_cnet --model-name=%s --in_dims='(img=%s)' --conv_fwd='(mode=rtc,write_op_sigs=1)'" %(model_dir,img) )

# just run with no args, will remove and then create op_sigs_full.txt:
# moskewcz@maaya:~/git_work/boda/run/tr3$ python ../../pysrc/cnn-prof-op-sigs-gen.py 
# .....

# moskewcz@maaya:~/git_work/boda/run/tr3$ wc op_sigs_full.txt 
#    429    429 104011 op_sigs_full.txt

#  notes: to filter for just Convolutions:
# moskewcz@maaya:~/git_work/boda/run/tr3$ grep type=Convolution op_sigs_full.txt  | wc
#    204     204   64918
