import os
runs = [("nin_imagenet", "NiN"), ("alexnet_ng_conv","AlexNet_no_groups"),("googlenet_conv","GoogLeNet"),]#("vgg_19","VGG-19")]
imgs = [1,5,20]

os.system( "rm op_sigs_full.txt" );
for model_dir,net_name in runs:
    for img in imgs:
        os.system( "boda run_cnet --model-name=%s --in_dims='(img=%s)' --conv_fwd='(mode=rtc,write_op_sigs=1)'" %(model_dir,img) )


