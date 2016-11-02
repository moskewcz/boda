
#note: to be run from a run directory at a depth like: boda/run/tr1, so that ../.. is the root of a boda WC. e.g:
# cd boda/run/tr1
# python ../../pysrc/boda-aa-fig-gen.py

#note: for safety, start with an empty directory, or at least remove
#all csv/png/pdf files before starting. that way, in case of errors,
#stale files won't be used. and actually read the output to look for
#errors!


import os

def run_cmds( cmds, fmt_data ):
    for cmd in cmds:
        fin_cmd = cmd % fmt_data
        print "RUN: "+fin_cmd
        os.system( fin_cmd )

class figgen_t( object ):
    def __init__( self, args ):
        self.args = args

        fmt_data = { "wis" : "--wisdom-in-fn=" + args.wis_fn, 
                     "cudnn_ref" : "--ref-tune='(use_be=nvrtc,use_culibs=1)'",
                     "s_img" : "--s-img=5" }
        # generate ocl/nvrtc comparison figure
        cmds = [
            "boda wis-ana %(wis)s %(s_img)s --s-plat='ocl.*TITAN' --csv-out-fn=out-tit-ocl-5-pom.csv --csv-res-tag='-ocl' --show-ref=0 --show-aom=0",
            "boda wis-ana %(wis)s %(s_img)s %(cudnn_ref)s --s-plat='nvrtc.*TITAN' --csv-out-fn=out-tit-nvrtc-5-pom.csv --csv-res-tag='-nvrtc' --show-ref=0 --show-aom=0",
            "python ../../pysrc/wis-plot.py out-tit-ocl-5-pom.csv out-tit-nvrtc-5-pom.csv --out-fn=titan-ocl-vs-nvrtc --out-fmt=pdf --title='OpenCL vs CUDA(nvrtc) Speed on NVIDIA Titan-X(Maxwell)'",
        ]
        run_cmds( cmds, fmt_data )
        # generate AMD tuning figure
        cmds = [
            "boda wis-ana %(wis)s %(s_img)s --s-plat='Fiji' --csv-out-fn=out-fiji.csv --show-ref=0",
            "python ../../pysrc/wis-plot.py out-fiji.csv --out-fn=fiji-tune --out-fmt=pdf --title='Tuned and Autotuned Speed on AMD R9-Nano(Fiji)'" ]
        run_cmds( cmds, fmt_data )
        # generate NVIDIA tuning figure
        cmds = [
            "boda wis-ana %(wis)s %(s_img)s %(cudnn_ref)s --s-plat='nvrtc.*TITAN' --csv-out-fn=out-titan.csv --show-ref=1",
            "python ../../pysrc/wis-plot.py out-titan.csv --out-fn=titan-tune --out-fmt=pdf --title='Tuned, Autotuned, and Reference(cuDNNv5) Speed on NVIDIA Titan-X(Maxwell)'" ]
        run_cmds( cmds, fmt_data )
        # generate all-plats figure
        cmds = [
            "boda wis-ana %(wis)s %(s_img)s %(cudnn_ref)s --s-plat='nvrtc.*TITAN' --csv-out-fn=all-plats-titan.csv --show-ref=0 --show-aom=0 --csv-res-tag='-titan'",
            "boda wis-ana %(wis)s %(s_img)s --s-plat='Fiji' --csv-out-fn=all-plats-fiji.csv --show-ref=0 --show-aom=0 --csv-res-tag='-R9'",
            "boda wis-ana %(wis)s %(s_img)s --s-plat='Adreno' --csv-out-fn=all-plats-SD820.csv --show-ref=0 --show-aom=0 --csv-res-tag='-SD820'",
            "python ../../pysrc/wis-plot.py all-plats-titan.csv all-plats-fiji.csv all-plats-SD820.csv --out-fn=all-plats --out-fmt=pdf --title='Autotuned Speed on NVIDIA Titan-X, AMD R9-Nano, and Qualcomm Snapdragon 820'" ]
        run_cmds( cmds, fmt_data )
        

        



import argparse
parser = argparse.ArgumentParser(description='generate list of figures for boda-aa paper.')
parser.add_argument('--wis-fn', metavar="FN", type=str, default="../../test/wisdom-merged.wis", help="filename of results database" )
args = parser.parse_args()

figgen_t( args )

