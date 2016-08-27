
# a refence conv: (type=Convolution,dims_vals=(biases=(out_chan=16),filts=(out_chan=16,in_chan=96,y=1,x=1),in=(img=3,chan=96,y=55,x=55),in_pad=(y=0,x=0),kern_sz=(y=1,x=1),out=(img=3,chan=16,y=55,x=55),stride=(y=1,x=1)),str_vals=(out_chans=16))

def emit_conv( tag, img, in_xy, kern_xy, stride, in_chan, out_chan ):
    #print batch_sz, in_xy, kern_xy, stride, in_chans, out_chans
    # FIXME: for now, we calculate out_xy here. maybe we should instead emit a pair of:
    # -- an operation at the higher level (without IO dims), 
    # -- and the input dims.
    # then we'd use a flow like the normal net-reading flow that would calculate the out dims (and full operation) from the input and operation (like a one-layer net).
    out_xy = [0,0]
    for d in [0,1]: 
        assert in_xy[d] > kern_xy[d]
        out_xy[d] = 1 + (in_xy[d]-kern_xy[d]) / stride[d]
    params = { 
        "tag":tag,
        "kx":kern_xy[0], "ky":kern_xy[1], 
        "img": img, 
        "out_chan": out_chan,
        "in_chan": in_chan,
        "sx":stride[0], "sy":stride[1], 
        "inx":in_xy[0], "iny":in_xy[1], 
        "outx":out_xy[0], "outy":out_xy[1], 
    }
    print ( "(str_vals=(type=Convolution),nda_vals=(biases=(dims=(out_chan=%(out_chan)s)),"
            "filts=(dims=(out_chan=%(out_chan)s,in_chan=%(in_chan)s,y=%(ky)s,x=%(kx)s)),"
            "in=(dims=(img=%(img)s,chan=%(in_chan)s,y=%(inx)s,x=%(iny)s)),"
            "in_pad=(dims=(y=0,x=0)),"
            "kern_sz=(dims=(y=%(ky)s,x=%(kx)s)),"
            "out=(dims=(img=%(img)s,chan=%(out_chan)s,y=%(outy)s,x=%(outx)s)),"
            "stride=(dims=(y=%(sy)s,x=%(sx)s)),"
            "out_chans=(tn=uint32_t,v=%(out_chan)s)))" % ( params ) )

def emit_conv_sweep():
    conv_ix = 0
    for batch_sz in [1, 2, 5, 10, 20]:
        for activX in [8, 9, 16, 17, 32, 33, 64, 65 ]: # activations X-dim
            for activY in [activX]:#[8, 9, 16, 17, 32, 33, 64, 65 ]: # activations Y-dim
                for filtX in [1, 2, 3, 4, 5]:
                    for filtY in [filtX]:#[1, 2, 3, 4, 5]:
                        for strideX in xrange(1, filtX+1):
                            for strideY in [strideX]:#in xrange(1, filtY+1):
                                for chans_in in [3, 4, 5, 8, 9, 16, 17, 32, 33, 128, 256, 512, 1024 ]:
                                    for chans_out in [3, 4, 5, 8, 9, 16, 17, 32, 33, 128, 256, 512, 1024 ]:
                                        tag = "op_"+str(conv_ix)
                                        conv_ix += 1
                                        emit_conv( tag, batch_sz, (activX, activY), (filtX, filtY), (strideX, strideY), 
                                                   chans_in, chans_out)

def emit_conv_small_sweep():
    conv_ix = 0
    for batch_sz in [1, 10]:
        for activX in [8, 17 ]: # activations X-dim
            for activY in [activX]: # activations Y-dim
                for filtX in [1, 3]:
                    for filtY in [filtX]:
                        for strideX in xrange(1, filtX):
                            for strideY in [strideX]:
                                for chans_in in [3, 33 ]:
                                    for chans_out in [3, 33 ]:
                                        tag = "op_"+str(conv_ix)
                                        conv_ix += 1
                                        emit_conv( tag, batch_sz, (activX, activY), (filtX, filtY), (strideX, strideY), 
                                                   chans_in, chans_out)


def emit_sgemm( M, N, K ):
    params = {"M":M,"N":N,"K":K,} 
    print ( "(type=sgemm,dims_vals=(a=(K=%(K)s,M=%(M)s),b=(K=%(K)s,N=%(N)s),c=(M=%(M)s,N=%(N)s)))" % ( params ) )


def emit_sgemm_sweep():
    for MNK in [32,64,128,256,384,512,768,1024,1536,2048,3072,4096,5120,6144,7168,8192,10240,12288]:
        emit_sgemm( MNK, MNK, MNK )


import argparse
parser = argparse.ArgumentParser(description='generate list of operations.')
parser.add_argument('--set', metavar="SETNAMESTR", type=str, default="conv_small", help="name of set to generate" )
args = parser.parse_args()

vars()["emit_"+args.set+"_sweep"]()

