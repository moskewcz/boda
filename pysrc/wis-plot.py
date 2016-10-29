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
        assert len(elp) == 4
        self.cols = ["varname","rts","flops"] #["varname","bxf","flops","ai","rts"]
        self.varname = "1"
        self.rts = float(elp[1])
        self.flops = float(elp[2])
        self.opinfo = elp[3]
        self.comp = None
        self.comp_rtc = float(elp[0]) 
    def __str__( self ):
        return " ".join( str(col)+"="+str(getattr(self,col)) for col in self.cols )


class varinfo( object ):
    def __init__( self, name, color, mark='o', mark_comp='d' ):
        self.name = name
        self.color = color
        self.mark = mark
        self.mark_comp = mark_comp
        self.art = plt.Line2D((0,0),(0,0), color=self.color, marker=self.mark, linestyle='')
        self.art_comp = plt.Line2D((0,0),(0,0), color=self.color, marker=self.mark_comp, linestyle='')
        self.num_use = 0
        self.num_use_comp = 0
    def clear_use( self ):
        self.num_use = 0
        self.num_use_comp = 0        
    def inc_use( self, is_comp ):
        if is_comp: 
            self.num_use_comp += 1 
        else:
            self.num_use += 1
    def get_mark( self, is_comp ): 
        return self.mark_comp if is_comp else self.mark

    def get_leg( self, leg_art, leg_lab ):
        verb_name = "\\verb|"+self.name+"|"
        if self.num_use:
            leg_art.append( self.art) 
            leg_lab.append( verb_name )
        if self.num_use_comp:
            leg_art.append( self.art_comp) 
            leg_lab.append( verb_name[:-1] + " (Comp)|" )
        self.clear_use()
        
vis = [ 
    varinfo( "conv", "cornflowerblue" ),
    varinfo( "conv_simd", "cornflowerblue" ),
    varinfo( "k1conv", "green" ),
    varinfo( "k1conv_simd", "green" ),
    varinfo( "tconv", "purple" ),
    varinfo( "cudnn_conv", "red" ),
]        
#vis_map = { vi.name:vi for vi in vis }
vis_map = { str(i):varinfo(str(i),"cornflowerblue") for i in range(20) }

def inc_comp( epts ):
    for ept in epts:
        yield ept
        if ept.comp: yield ept.comp

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
        self.epts_comp = []
        read_eff_file( self.epts, self.args.eff_fn )
        if self.args.eff_comp_fn:
            read_eff_file( self.epts_comp, self.args.eff_comp_fn )
            assert len(self.epts) == len(self.epts_comp)
            for ept,ept_comp in zip(self.epts,self.epts_comp):
                assert ept.opinfo == ept_comp.opinfo
                ept.comp = ept_comp
        self.do_plots()
        if self.args.do_zooms:
            for zl in [1,2]:
                self.args.out_fn += "-zoom"
                max_flops = max( ept.flops for ept in self.epts )
                self.epts = [ ept for ept in self.epts if ept.flops < (max_flops/10.0) ]
                self.do_plots()

    def skip_plot_check_flops_vs_time( self, ept ):
        return 0
        if not ept.comp: return 0 # no comp? if so, never skip.
        delta = abs( ept.rts - ept.comp.rts )
        rel_delta = delta * 2.0 / (ept.rts + ept.comp.rts)
        # if rel_delta < self.args.min_rel_delta_to_show: return 1 # we're really trying to show varaint difference, so skip this check
        if ept.varname == ept.comp.varname: return 1 # FIXME: skip when comp is same varaint. not right in general, but okay for now
        # FIXME: a few data points have the same variant, but sig. diff runtimes. there is certainly some noise in the runtimes, or the code might have shifted a bit between the two runs, or it's possible the tuning params were a little different between the two runs. for now, we'll skip such points, but we should investigate more. 
        return 0

    def plot_flops_vs_time_pt( self, ax, ept, is_comp ):
        vi = vis_map[ept.varname]
        vi.inc_use( is_comp )
        x,y = math.log(ept.flops,10), math.log(ept.rts,10)
        ax.plot(x, y, color=vi.color, markersize=4, alpha=.7, marker=vi.get_mark(is_comp), linestyle=' ' )
        return x,y

    def plot_fps_vs_ai_pt( self, ax, ept, is_comp ):
        vi = vis_map[ept.varname]
        vi.inc_use( is_comp )
        x = ept.ai
        y = ept.flops/ept.rts
        ax.plot( x,y, color=vi.color, markersize=2*max(1,math.log(ept.flops,10)-6), alpha=.7, marker=vi.get_mark(is_comp), linestyle=' ' )
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
        x = [ math.log(ept.flops,10) for ept in inc_comp(self.epts) ]
        y = [ math.log(ept.rts,10) for ept in inc_comp(self.epts) ]
        self.set_bnds( ax, x, y )

        # print matplotlib.lines.Line2D.filled_markers 
        # --> (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
        for ept in self.epts:
            x,y = self.plot_flops_vs_time_pt( ax, ept, 0 )
            if ept.comp:
                xc,yc = self.plot_flops_vs_time_pt( ax, ept.comp, 1 )
                ax.plot( [x,xc], [y,yc], linewidth=0.5, color='black' )

        leg_art = []; leg_lab = []
        for vi in vis: vi.get_leg( leg_art, leg_lab )
        legend = ax.legend(leg_art,leg_lab,loc='lower right', shadow=True, fontsize='small',numpoints=1,ncol=1)
        legend.get_frame().set_facecolor('#eeddcc')

        max_fps = max( ept.flops/ept.rts for ept in inc_comp(self.epts) )
        log10_max_fps = int(math.ceil(math.log(max_fps,10)))
        if 1:
            fps_bnd = 10**log10_max_fps
            self.add_fps_line( ax, fps_bnd / 10.0 )
            self.add_fps_line( ax, fps_bnd / 5.0 )
            self.add_fps_line( ax, fps_bnd / 2.0 )
            self.add_fps_line( ax, fps_bnd )

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

    def add_fps_line( self, ax, fps ): self.add_fps_line_log( ax, fps )

    def add_fps_line_lin( self, ax, fps ):
        #Peak performance line and text
        x = [self.x_min,(self.x_min+self.x_max)*0.5,self.x_max]
        y = [ v/fps for v in x ]
        y_mid = (self.y_min+self.y_max)/2
        if y[1] > y_mid: # high slope case; use target y val
            y[1] = y_mid
            x[1] = y[1]*fps
        ax.plot(x,y, linewidth=1.0, color='black', linestyle=':' )
        label_string = "%.1fGF/s" % (fps/1e9)
        rot=np.arctan(y[1]/x[1]*self.axis_aspect) * 180 / np.pi
        ax.text(x[1], y[1], label_string, fontsize=8, rotation=rot, ha="left", va="bottom")

    def add_fps_line_log( self, ax, fps ):
        #Peak performance line and text
        x = [self.x_min,self.x_min*0.2+self.x_max*0.8,self.x_max]
        y = [ v - math.log(fps,10) for v in x ]
        y_mid = self.y_min*0.2+self.y_max*0.8
        if y[1] > y_mid: # high slope case; use target y val
            y[1] = y_mid
            x[1] = y[1] + math.log(fps,10)
        ax.plot(x,y, linewidth=1.0, color='black', linestyle=':' )
        label_string = "%.1fGF/s" % (fps/1e9)
        rot=np.arctan(self.data_aspect) * 180 / np.pi
        ax.text(x[1], y[1], label_string, fontsize=12, rotation=rot, ha="left", va="bottom")



import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('--eff-fn', metavar="FN", type=str, default="out.csv", help="filename of eff values in csv format" )
parser.add_argument('--eff-comp-fn', metavar="FN", type=str, default="", help="filename of eff values in latex table format for comparison to those from the file specified by --eff-fn" )
parser.add_argument('--out-fn', metavar="FN", type=str, default="eff", help="base filename of output plot image" )
parser.add_argument('--out-fmt', metavar="EXT", type=str, default="png", help="extention/format for output plot image" )
parser.add_argument('--do-zooms', metavar="BOOL", type=bool, default=0, help="if true, output zoomed and 2X zoomed graphs" )
parser.add_argument('--min-rel-delta-to-show', metavar="FLOAT", type=float, default=0.05, help="if true, skip showing points where delta/avg is < this value in comparison mode" )
args = parser.parse_args()
ep = EffPlot(args)

# example command lines for generating inputs to this script:

# boda on titan-X, optimized variants enabled
# boda cnn_op_info --cnn-func-sigs-fn='%(boda_test_dir)'/conv-ops-1-5-20-nin-alex-gn.txt --op-eff-tab-fn=conv-1-5-20-nin-alex-gn-titanX-boda.raw  --rtc='(be=nvrtc)' --gen-data='(type=foo,str_vals=(vi=0.0f,mode=5))' --op-tune='(tconv=1,k1conv=1)' --rtc-comp='(be=nvrtc)'  --max-err=10 --show-rtc-calls=1 --mad-toler=3e-3 --print-format=1 --inc-op-info-in-eff=1

# run on SD820, optimizations enabled, no comparison:
# export SD820_RTC="rtc=(be=ipc,remote_rtc=(be=ocl,gen_src=1,gen_src_output_dir=/data/local/rtc-gen-src),spawn_str=adb shell LD_LIBRARY_PATH=/data/local/lib /data/local/bin/boda,spawn_shell_escape_args=1,boda_parent_addr=tcp:10.0.0.100:12791)"
# export OP_TUNE="op_tune=(use_culibs=0,MNt=8:8,MNb=16:16,k1conv=1,tconv=0,Kb=1,vw=8,use_local_mem=2)"
# boda cnn_op_info --cnn-func-sigs-fn='%(boda_test_dir)'/conv-ops-1-5-20-nin-alex-gn.txt --op-eff-tab-fn=conv-1-5-20-nin-alex-gn-SD820-boda.raw --"${SD820_RTC}" --"${OP_TUNE}" --show-rtc-calls=1 --peak-flops=320e9 --print-format=1 --inc-op-info-in-eff=1
