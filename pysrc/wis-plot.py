from matplotlib import rc
rc('text', usetex=True) # this is if you want to use latex to print text. If you do you can create strings that go on labels or titles like this for example (with an r in front): r"$n=$ " + str(int(n))
from numpy import *
from pylab import *
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as lns
from scipy import stats
from matplotlib.patches import Polygon, Circle
import matplotlib.font_manager as fm

nan = float("nan")
from math import isnan

def zero_nan(v): return 0.0 if (v == nan) else v

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

class EffPt( object ):
    def __init__( self, opinfo, flops ):
        self.opinfo = opinfo
        self.flops = flops
        self.rtss = {}

    def __repr__( self ):
        return "%s %s %s" % (self.opinfo, self.flops, self.rtss)
    def non_nan_rts( self ): return [rts for rts in self.rtss.itervalues() if not isnan(rts)]


class varinfo( object ):
    def __init__( self, name, color, mark='s' ):
        self.name = name
        self.color = color
        self.mark = mark
        self.art = plt.Line2D((0,0),(0,0), color=self.color, marker=self.mark, linestyle='')
    def get_mark( self ): return self.mark

    def get_leg( self, leg_art, leg_lab ):
        verb_name = "\\verb|"+self.name+"|"
        leg_art.append( self.art) 
        leg_lab.append( verb_name )

colors = [ "cornflowerblue", "green", "red","black","yellow","purple" ]
vis = [ ]

def read_eff_file( epts, fn ):
    els = open( fn ).readlines()
    cols = [el.strip() for el in els[0].split(" ")]
    for col in cols[2:]: vis.append( varinfo( col, colors[len(vis)] ) )

    assert cols[0] == "OP"
    assert cols[1] == "FLOPS"

    for el in els[1:]:
        elps = el.split(" ")
        elps = [ elp.strip() for elp in elps ]
        assert len(elps) == len(cols)
        opinfo = elps[0]
        flops = float(elps[1])
    
        ept = epts.setdefault( opinfo, EffPt( opinfo, flops ) )
        assert ept.flops == flops
        for elp,col in zip( elps[2:], cols[2:] ):
            assert not (col in ept.rtss)
            ept.rtss[col] = float(elp)
        
class EffPlot( object ):
    def __init__( self, args ):
        self.args = args
        self.epts = {}
        for fn in self.args.eff_fn:
            read_eff_file( self.epts, fn )
        self.do_plots()

    def do_plots( self ):
        # flops vs runtime plot with 60GF/s line
        background_color =(0.85,0.85,0.85) #'#C0C0C0'    
        grid_color = 'white' #FAFAF7'
        rc('axes', facecolor = background_color)
        rc('axes', edgecolor = grid_color)
        rc('axes', linewidth = 1.2)
        rc('axes', grid = True )
        rc('axes', axisbelow = True)
        rc('grid',color = grid_color)
        rc('grid',linestyle='-' )
        rc('grid',linewidth=0.7 )
        #rc('xtick.major',size =0 )
        #rc('xtick.minor',size =0 )
        #rc('ytick.major',size =0 )
        #rc('ytick.minor',size =0 )

        # convert epts map to list; sort data by flops
        self.epts = self.epts.values()
        self.epts.sort( key=lambda x: x.flops )
        for ix,ept in enumerate(self.epts): ept.ix = ix

        all_rts = []
        for ept in self.epts: all_rts.extend( ept.non_nan_rts() )
        floor_log10_min_v = math.floor(math.log(min(all_rts),10))
        ceil_log10_max_v = math.ceil(math.log(max(all_rts),10))
        #print floor_log10_min_v, ceil_log10_max_v

        fig = plt.figure()
        plt.yscale('log', nonposy='clip')
        ax = fig.add_subplot(111)
        ax.xaxis.grid(0)
        ax.xaxis.set_ticks([])
        #ax.set_autoscaley_on(True)
        ax.set_ylim([10**floor_log10_min_v,10**ceil_log10_max_v])
        #formatting:
        ax.set_title(self.args.title,fontsize=12,fontweight='bold')
        ax.set_xlabel("Individual Convolutions, sorted by \\#-of-FLOPS", fontsize=12) # ,fontproperties = font)
        ax.set_ylabel("Runtime (seconds)", fontsize=12) # ,fontproperties = font)

        # print matplotlib.lines.Line2D.filled_markers 
        # --> (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
        num_bars = 1
        width = 1.0 / (len(vis) + 1 )
        offset = 0.0
        for vi in vis:
            vi_data = [ (ept.ix, ept.rtss[vi.name]) for ept in self.epts if ( not isnan( ept.rtss.get(vi.name,nan) ) ) ]
            vi_y = [ d[1] for d in vi_data ]
            ixs = [ d[0] for d in vi_data ]
            rects = ax.bar(np.array(ixs) + offset, vi_y, width, log=True, color=vi.color, linewidth=0 ) # note: output rects unused
            offset += width

        leg_art = []; leg_lab = []
        for vi in vis: vi.get_leg( leg_art, leg_lab )
        legend = ax.legend(leg_art,leg_lab,loc='lower right', shadow=True, fontsize='small',numpoints=1,ncol=1)
        legend.get_frame().set_facecolor('#eeddcc')

        #ax.autoscale_view()
        #ax.plot()
        fig.savefig( self.args.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')



import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('eff_fn', nargs='*', metavar="FN", type=str, default="out.csv", help="filename of eff values in csv format" )
parser.add_argument('--out-fn', metavar="FN", type=str, default="eff", help="base filename of output plot image" )
parser.add_argument('--out-fmt', metavar="EXT", type=str, default="png", help="extention/format for output plot image" )
parser.add_argument('--title', metavar="STR", type=str, default="Per-Convolution Runtime (seconds) [log scale]", help="plot title" )

args = parser.parse_args()
ep = EffPlot(args)
