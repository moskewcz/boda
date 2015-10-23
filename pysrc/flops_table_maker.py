import os
runs = [("nin_imagenet", "NiN"), ("alexnet_ng_conv","AlexNet_no_groups"),("figln","GoogLeNet"),("vgg_19","VGG-19")]

out = open("tab.tex","w")
header = "%-20s & %-20s & %-20s & %-20s & %-20s\\hline\n" % ( "DNN architecture", "data_size", "weight_size", "data/weight ratio", "F+B flops/img" )
out.write( header )
out.close() # appended to be os.system calls below

for model_dir,net_name in runs:
    os.system( "boda cnet_ana --in-model=%s --print-ops=1 --in-sz=224  && python ../../pysrc/flops.py  --print-tex-table-entry=1 --per-layer=1 --backward=1 --net-name=%s | grep hline >> tab.tex" %(model_dir,net_name) )

