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

def zero_nan(v): return 0.0 if (v == nan) else v

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

class EffPt( object ):
    def __init__( self, elp, cols ):
        assert len(elp) == len(cols)
        assert cols[0] == "OP"
        assert cols[1] == "FLOPS"
        self.cols = ["opinfo","flops"]
        self.data_cols = cols[2:]
        self.opinfo = elp[0]
        self.flops = float(elp[1])
        self.rtss = [ float(p) for p in elp[2:] ]

    def __repr__( self ):
        return " ".join( str(col)+"="+str(getattr(self,col)) for col in self.cols ) + " " + str(self.rtss)
    def max_rts( self ):
        return max(zero_nan(rts) for rts in self.rtss)


class varinfo( object ):
    def __init__( self, name, dix, color, mark='s' ):
        self.name = name
        self.dix = dix # data index inside EffPt
        self.color = color
        self.mark = mark
        self.art = plt.Line2D((0,0),(0,0), color=self.color, marker=self.mark, linestyle='')
    def get_mark( self ): return self.mark

    def get_leg( self, leg_art, leg_lab ):
        verb_name = "\\verb|"+self.name+"|"
        leg_art.append( self.art) 
        leg_lab.append( verb_name )

colors = [ "cornflowerblue", "green", "red" ]
vis = [ ]

def read_eff_file( epts, fn ):
    els = open( fn ).readlines()
    cols = [el.strip() for el in els[0].split(" ")]
    for dix,col in enumerate(cols[2:]):
        vis.append( varinfo( col, dix, colors[dix] ) )
    for el in els[1:]:
        elps = el.split(" ")
        elps = [ elp.strip() for elp in elps ]
        epts.append( EffPt( elps, cols ) )

def adj_tick_lab( lab ): 
    lt = lab.get_text()
    if not lt: return "" 
    if lt[0] == "$": lt = lt[1:-1]
    neg = 1.0
    if lt[0] == u'\u2212': lt = lt[1:]; neg = -1.0
    return "$%s$" % latex_float(10**(neg*float(lt)))

class EffPlot( object ):
    def __init__( self, args ):
        self.args = args
        self.epts = []
        read_eff_file( self.epts, self.args.eff_fn )
        self.do_plots()

    def skip_plot_check_flops_vs_time( self, ept ):
        return 0

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

        # filter data based on skip check
        self.epts = [ ept for ept in self.epts if not self.skip_plot_check_flops_vs_time( ept ) ]

        # sort data by flops
        self.epts.sort( key=lambda x: x.flops )

        ixs = [ ix for (ix,ept) in enumerate(self.epts) ]
        
        fig = plt.figure()
        plt.yscale('log', nonposy='clip')
        ax = fig.add_subplot(111)
        ax.xaxis.grid(0)
        ax.xaxis.set_ticks([])
        #rc('xtick.major',size =0 )
        #rc('xtick.minor',size =0 )

        #formatting:
        ax.set_title(self.args.title,fontsize=12,fontweight='bold')
        ax.set_xlabel("Convolution index, sorted by \\#-of-FLOPS", fontsize=12) # ,fontproperties = font)
        ax.set_ylabel("RUNTIME (seconds)", fontsize=12) # ,fontproperties = font)

        #x = [ math.log(ept.flops,10) for ept in self.epts ]
        #x = [ math.log(ept.flops,10) for ept in self.epts ]
        # y = [ ept.max_rts() for ept in self.epts ]
        # self.set_bnds( ax, ixs, y )

        # print matplotlib.lines.Line2D.filled_markers 
        # --> (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
        num_bars = 1
        width = 1.0 / (len(vis) + 1 )
        offset = 0.0
        for vi in vis:
            vi_y = [ ept.rtss[vi.dix] for ept in self.epts ]
            rects = ax.bar(np.array(ixs) + offset, vi_y, width, log=True, color=vi.color, linewidth=0 ) # note: output rects unused
            offset += width


        leg_art = []; leg_lab = []
        for vi in vis: vi.get_leg( leg_art, leg_lab )
        legend = ax.legend(leg_art,leg_lab,loc='lower right', shadow=True, fontsize='small',numpoints=1,ncol=1)
        legend.get_frame().set_facecolor('#eeddcc')

        fig.savefig( self.args.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')

        # ai vs GF/s plot
    def set_bnds( self, ax, x, y ):
        self.x_min = min(x)
        self.x_max = max(x)*1.05
        self.y_min = min(y)
        self.y_max = max(y)*1.05
        ax.axis([self.x_min,self.x_max,self.y_min,self.y_max])



import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('--eff-fn', metavar="FN", type=str, default="out.csv", help="filename of eff values in csv format" )
parser.add_argument('--eff-comp-fn', metavar="FN", type=str, default="", help="filename of eff values in latex table format for comparison to those from the file specified by --eff-fn" )
parser.add_argument('--out-fn', metavar="FN", type=str, default="eff", help="base filename of output plot image" )
parser.add_argument('--out-fmt', metavar="EXT", type=str, default="png", help="extention/format for output plot image" )
parser.add_argument('--title', metavar="STR", type=str, default="Per-Convolution Runtime (seconds) [log scale]", help="plot title" )

args = parser.parse_args()
ep = EffPlot(args)
