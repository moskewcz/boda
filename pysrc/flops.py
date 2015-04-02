# pretty printing stuff. factor out somewhere?
def pp_val_part( v, force ):
    if v < 10: return "%.2f" % v
    if v < 100: return "%.1f" % v
    if (v < 1000) or force: return "%.0f" % v
    return None
def pp_val( v ): # pretty-print flops
    ret = pp_val_part( v, 0 )
    exp = 0
    while ret is None:
        v /= 1000.0
        exp += 1
        ret = pp_val_part( v, exp == 5 )
    if exp == 0: return ret
    return ret+"KMGTP"[exp - 1]
def pp_flops( v ): return pp_val( v ) + " FLOPS"
def pp_bytes( v ): return pp_val( v ) + " BYTES"
def pp_fpb( v ): return pp_val( v ) + " FLOPS/BYTE"
def pp_fps( v ): return pp_val( v ) + " FLOPS/SEC"
def pp_fpspw( v ): return pp_val( v ) + " FLOPS/SEC/WATT"
def pp_joules( v ): return pp_val( v ) + " JOULES"

import operator

# cnet flop-calculating ops classes
class NDA( object ): 
    def __init__( self, name, *args ):
        self.name = name
        if len(args) == 4:
            self.num = args[0]
            self.chan = args[1]
            self.y = args[2]
            self.x = args[3]
        else:
            self.dims = args
    def dims_prod( self ): 
        if hasattr(self,"dims"): return reduce(operator.mul, self.dims, 1)
        return self.num*self.chan*self.y*self.x

class Net( object ):
    def __init__( self ):
        self.tot_forward_flops = 0
        self.tot_forward_bytes = 0
        self.tot_backward_flops = 0
        self.tot_backward_bytes = 0

    def print_stats( self ):
        args = self.args
        print ""
        print "inputs: NUM_IMGS=%s RUNTIME=%s s POWER=%s W" % (args.num_imgs, args.runtime, args.power )
        print "TOTAL_ENERGY",pp_joules(args.power*args.runtime)

        fb_str = "FORWARD"
        flops = self.tot_forward_flops
        bytes_ = self.tot_forward_bytes
        if args.backward:
            fb_str = "FORWARD_BACKWARD"
            flops += self.tot_backward_flops
            bytes_ += self.tot_backward_bytes
        print "TOTAL_%s_FLOPS %s" % ( fb_str, pp_flops(flops) )
        print "TOTAL_%s_FLOPS_PER_SEC %s" % ( fb_str, pp_fps(flops/args.runtime) )
        print "TOTAL_%s_FLOPS_PER_SEC_PER_WATT %s" % ( fb_str, pp_fpspw(flops/args.runtime/args.power) )

        print "TOTAL_%s_IO_BYTES %s" % ( fb_str, pp_bytes(bytes_))
        print "TOTAL_%s_ARITH_INTENSITY %s" % ( fb_str, pp_fpb(flops / bytes_) )



net = Net()

class Convolution( object ): 
    def __init__( self, name, bots, tops, filts, biases, in_pad, stride ): 
        # note: ignores in_pad and stride, but they sort-of aren't
        # needed since the output size is calculated using them. we
        # could use them as a check here, but that would require
        # duplicating the in-out calculation code?
        global net
        assert len(bots) == 1
        assert len(tops) == 1
        bot = bots[0]
        top = tops[0]
        in_pels = bot.dims_prod()
        out_pels = top.dims_prod()
        forward_bytes = (in_pels + out_pels + filts.dims_prod() + biases.dims_prod()) * 4
        backward_bytes = (in_pels*2 + out_pels + filts.dims_prod()*2 + biases.dims_prod()*2) * 4

        net.tot_forward_bytes += forward_bytes
        net.tot_backward_bytes += backward_bytes
            
        assert bot.chan == filts.chan # filt.chan is (or should be) input chans
        assert top.chan == filts.num # filt.num is (or should be) output chans
        # note: top.{x,y} should be = ( bot.{x,y} + pad ) / stride   (ceild/floord correctly)
        forward_flops = out_pels * filts.x * filts.y * filts.chan * 2
        grad_inner_dim = out_pels / filts.num # aka number of input patches
        assert grad_inner_dim == top.num*top.x*top.y
        back_grad_flops = filts.dims_prod() * grad_inner_dim * 2 # grad is same size as filts
        diff_inner_dim = filts.num
        # diff ends up as the same size as input but is reduced from a temp of size im2col(input).
        back_diff_flops = (filts.chan*filts.x*filts.y)*diff_inner_dim*grad_inner_dim * 2  # as: (M) * N * K

        net.tot_forward_flops += forward_flops
        net.tot_backward_flops += back_diff_flops + back_grad_flops

        if net.args.per_layer:
            print name,
            print " FORWARD",pp_flops(forward_flops),
            if net.args.backward:
                print " --- BACK_GRAD",pp_flops(back_grad_flops),
                print " --- BACK_DIFF",pp_flops(back_diff_flops),
            print " FORWARD_BYTES",pp_bytes(forward_bytes),
            if net.args.backward:
                print " BACKWARD_BYTES",pp_bytes(backward_bytes),
            print ""

# FIXME: in boda output, the ops/nodes of IP layers are printed out as if it
# they were conv layers ... not ideal, since NDAs don't match caffe iface
# for innerproduct. hmm.
InnerProduct=Convolution 

class Pooling( object ): 
    def __init__( self, **kwargs ): self.opts = kwargs
class LRN( object ): 
    def __init__( self, **kwargs ): self.opts = kwargs
class Concat( object ): 
    def __init__( self, **kwargs ): self.opts = kwargs
class ReLU( object ): 
    def __init__( self, **kwargs ): self.opts = kwargs
class Dropout( object ): 
    def __init__( self, **kwargs ): self.opts = kwargs

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--net-fn', metavar="FN", type=str, default="out.py", help="filename of network-definition python script" )
parser.add_argument('--num-imgs', metavar='N', type=int, default=0, help='an integer for the accumulator')
parser.add_argument('--runtime', metavar='SECONDS', type=float, default=0, help='time taken for power/energy calculations')
parser.add_argument('--power', metavar='WATTS', type=float, default=0, help='average power used over runtime')
parser.add_argument('--backward', metavar='BOOL', type=int, default=1, help='1:forward+backward; 0:only forward')
parser.add_argument('--per-layer', metavar='BOOL', type=int, default=0, help='1:print per-layer info; 0:only summary')
args = parser.parse_args()
net.args = args
# set num_img and source cnet decl
num_img = args.num_imgs
execfile( args.net_fn )

net.print_stats()
