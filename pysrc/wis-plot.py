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

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

class EffPt( object ):
    def __init__( self, elp ):
        assert len(elp) == 5
        self.cols = ["varname","rts","flops"] #["varname","bxf","flops","ai","rts"]
        self.rts = float(elp[1])
        self.flops = float(elp[3])
        self.opinfo = elp[4]
        self.comp_rtc = float(elp[0]) 
        self.ref_rtc = float(elp[2]) 
    def __str__( self ):
        return " ".join( str(col)+"="+str(getattr(self,col)) for col in self.cols )


class varinfo( object ):
    def __init__( self, name, color, mark='o', mark_comp='d' ):
        self.name = name
        self.color = color
        self.mark = mark
        self.art = plt.Line2D((0,0),(0,0), color=self.color, marker=self.mark, linestyle='')
    def get_mark( self ): return self.mark

    def get_leg( self, leg_art, leg_lab ):
        verb_name = "\\verb|"+self.name+"|"
        leg_art.append( self.art) 
        leg_lab.append( verb_name )
        
vis = [ 
    varinfo( "AOM", "cornflowerblue" ),
    varinfo( "POM", "green" ),
    varinfo( "REF", "red" ),
]        
vis_map = { vi.name:vi for vi in vis }

def read_eff_file( epts, fn ):
    els = open( fn ).readlines()
    for el in els:
        elps = el.split(" ")
        elps = [ elp.strip() for elp in elps ]
        #print len(elps), elps
        epts.append( EffPt( elps ) )
        if math.isnan(epts[-1].rts): epts.pop()

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


    def plot_flops_vs_time_pt( self, ax, ept ):
        vi = vis_map["AOM"]
        x,y = math.log(ept.flops,10), math.log(ept.rts,10)
        ax.plot(x, y, color=vi.color, markersize=4, alpha=.7, marker=vi.get_mark(), linestyle=' ' )
        return x,y

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
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #formatting:
        ax.set_title("RUNTIME (seconds) vs \\#-of-FLOPS [log/log scale]",fontsize=12,fontweight='bold')
        ax.set_xlabel("\\#-of-FLOPS", fontsize=12) # ,fontproperties = font)
        ax.set_ylabel("RUNTIME (seconds)", fontsize=12) # ,fontproperties = font)
        x = [ math.log(ept.flops,10) for ept in self.epts ]
        y = [ math.log(ept.rts,10) for ept in self.epts ]
        self.set_bnds( ax, x, y )

        # print matplotlib.lines.Line2D.filled_markers 
        # --> (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
        for ept in self.epts: x,y = self.plot_flops_vs_time_pt( ax, ept )

        leg_art = []; leg_lab = []
        for vi in vis: vi.get_leg( leg_art, leg_lab )
        legend = ax.legend(leg_art,leg_lab,loc='lower right', shadow=True, fontsize='small',numpoints=1,ncol=1)
        legend.get_frame().set_facecolor('#eeddcc')

        self.adj_ticks(ax,fig)

        fig.savefig( self.args.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')

        # ai vs GF/s plot
    def set_bnds( self, ax, x, y ):
        self.x_min = min(x)
        self.x_max = max(x)*1.05
        self.y_min = min(y)
        self.y_max = max(y)*1.05
        ax.axis([self.x_min,self.x_max,self.y_min,self.y_max])

        self.data_aspect = float(self.x_max - self.x_min ) / (self.y_max - self.y_min)
        #self.axis_aspect_rat = .618
        self.axis_aspect_rat = 1
        self.axis_aspect = self.axis_aspect_rat * self.data_aspect
        ax.set_aspect(self.axis_aspect)
        
    def adj_ticks( self, ax, fig ):
        fig.canvas.draw()
        tls = ax.get_xticklabels()
        tls = [ adj_tick_lab(lab) for lab in tls ]
        ax.set_xticklabels( tls )

        tls = ax.get_yticklabels()
        tls = [ adj_tick_lab(lab) for lab in tls ]
        ax.set_yticklabels( tls )



import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('--eff-fn', metavar="FN", type=str, default="out.csv", help="filename of eff values in csv format" )
parser.add_argument('--eff-comp-fn', metavar="FN", type=str, default="", help="filename of eff values in latex table format for comparison to those from the file specified by --eff-fn" )
parser.add_argument('--out-fn', metavar="FN", type=str, default="eff", help="base filename of output plot image" )
parser.add_argument('--out-fmt', metavar="EXT", type=str, default="png", help="extention/format for output plot image" )
args = parser.parse_args()
ep = EffPlot(args)

# example command lines for generating inputs to this script:

# boda on titan-X, optimized variants enabled
# boda cnn_op_info --cnn-func-sigs-fn='%(boda_test_dir)'/conv-ops-1-5-20-nin-alex-gn.txt --op-eff-tab-fn=conv-1-5-20-nin-alex-gn-titanX-boda.raw  --rtc='(be=nvrtc)' --gen-data='(type=foo,str_vals=(vi=0.0f,mode=5))' --op-tune='(tconv=1,k1conv=1)' --rtc-comp='(be=nvrtc)'  --max-err=10 --show-rtc-calls=1 --mad-toler=3e-3 --print-format=1 --inc-op-info-in-eff=1

# run on SD820, optimizations enabled, no comparison:
# export SD820_RTC="rtc=(be=ipc,remote_rtc=(be=ocl,gen_src=1,gen_src_output_dir=/data/local/rtc-gen-src),spawn_str=adb shell LD_LIBRARY_PATH=/data/local/lib /data/local/bin/boda,spawn_shell_escape_args=1,boda_parent_addr=tcp:10.0.0.100:12791)"
# export OP_TUNE="op_tune=(use_culibs=0,MNt=8:8,MNb=16:16,k1conv=1,tconv=0,Kb=1,vw=8,use_local_mem=2)"
# boda cnn_op_info --cnn-func-sigs-fn='%(boda_test_dir)'/conv-ops-1-5-20-nin-alex-gn.txt --op-eff-tab-fn=conv-1-5-20-nin-alex-gn-SD820-boda.raw --"${SD820_RTC}" --"${OP_TUNE}" --show-rtc-calls=1 --peak-flops=320e9 --print-format=1 --inc-op-info-in-eff=1
