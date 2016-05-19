from matplotlib import rc
#rc('text', usetex=True) # this is if you want to use latex to print text. If you do you can create strings that go on labels or titles like this for example (with an r in front): r"$n=$ " + str(int(n))
from numpy import *
from pylab import *
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as lns
from scipy import stats
from matplotlib.patches import Polygon, Circle
import matplotlib.font_manager as fm

class EffPt( object ):
    def __init__( self, elp ):
        self.cols = ["varname","bxf","flops","ai","rts"]
        self.varname = elp[7][6:-1]
        self.bxf = float(elp[8])
        self.flops = float(elp[9])
        self.ai = float(elp[10])
        self.rts = float(elp[14])
    def __str__( self ):
        return " ".join( str(col)+"="+str(getattr(self,col)) for col in self.cols )


class varinfo( object ):
    def __init__( self, name, color, art ):
        self.name = name
        self.color = color
        self.art = art

vis = [ varinfo( "conv_simd", "cornflowerblue",  plt.Line2D((0,0),(0,0), color="cornflowerblue", marker='o', linestyle='') ),
        varinfo( "k1conv_simd", "green",  plt.Line2D((0,0),(0,0), color="green", marker='o', linestyle='') ) ]        

class EffPlot( object ):
    def __init__( self, args ):
        self.args = args
        self.epts = []
        els = open( self.args.eff_fn ).readlines()
        for el in els:
            elps = el.split()
            elps = [ elp for elp in elps if (elp not in ["$","&","\\dx","\\\\"]) ]
            self.epts.append( EffPt( elps ) )
        self.do_plots()
        for zl in []: # [1,2]:
            self.args.out_fn += "-zoom"
            max_flops = max( ept.flops for ept in self.epts )
            self.epts = [ ept for ept in self.epts if ept.flops < (max_flops/10.0) ]
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
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #formatting:
        ax.set_title("RUNTIME vs #-of-FLOPS",fontsize=14,fontweight='bold')
        ax.set_xlabel("#-of-FLOPS", fontsize=12) # ,fontproperties = font)
        ax.set_ylabel("RUNTIME", fontsize=12) # ,fontproperties = font)

        x = [ ept.flops for ept in self.epts ]
        y = [ ept.rts for ept in self.epts ]
        self.set_bnds( ax, x, y )

        # print matplotlib.lines.Line2D.filled_markers 
        # --> (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
        for vi in vis:
            x = [ ept.flops for ept in self.epts if ept.varname == vi.name ]
            y = [ ept.rts for ept in self.epts if ept.varname == vi.name ]
            ax.plot(x, y, color=vi.color, markersize=4, alpha=.7, marker='o', linestyle=' ' )

        leg_art = [vi.art for vi in vis]
        leg_lab = [vi.name for vi in vis]
    
        legend = ax.legend(leg_art,leg_lab,loc='lower right', shadow=True, fontsize='small',numpoints=1,ncol=1)
        legend.get_frame().set_facecolor('#eeddcc')

        self.add_fps_line( ax, .2, 10*1e9 )
        self.add_fps_line( ax, .4, 20*1e9 )
        self.add_fps_line( ax, .5, 50*1e9 )
        self.add_fps_line( ax, .5, 100*1e9 )
        fig.savefig( self.args.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')


        fig = plt.figure()
        ax = fig.add_subplot(111)
        #formatting:
        ax.set_title("GF/s vs Arithmetic Intensity; size == log10(#-of-FLOPS)",fontsize=12,fontweight='bold')
        ax.set_xlabel("Arithmetic Intensity", fontsize=12) # ,fontproperties = font)
        ax.set_ylabel("GF/s", fontsize=12) # ,fontproperties = font)

        x = [ ept.ai for ept in self.epts ]
        y = [ ept.flops/ept.rts for ept in self.epts ]
        self.set_bnds( ax, x, y )

        # print matplotlib.lines.Line2D.filled_markers 
        # --> (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
        for vi in vis:
            x = [ ept.ai for ept in self.epts if ept.varname == vi.name ]
            y = [ ept.flops/ept.rts for ept in self.epts if ept.varname == vi.name ]
            #ax.plot(x, y, color=vi.color, markersize=msz, alpha=.7, marker='o', linestyle=' ' )
            #msz = [ math.log(ept.flops) for ept in self.epts if ept.varname == vi.name ]
            for ept in self.epts:
                if ept.varname != vi.name: continue
                ax.plot( ept.ai, ept.flops/ept.rts, color=vi.color, 
                         markersize=2*max(1,math.log(ept.flops,10)-6), alpha=.7, marker='o', linestyle=' ' )

        leg_art = [vi.art for vi in vis]
        leg_lab = [vi.name for vi in vis]
    
        legend = ax.legend(leg_art,leg_lab,loc='lower right', shadow=True, fontsize='small',numpoints=1,ncol=1)
        legend.get_frame().set_facecolor('#eeddcc')

        #self.add_fps_line( ax, .2, 10*1e9 )
        #self.add_fps_line( ax, .4, 20*1e9 )
        #self.add_fps_line( ax, .5, 50*1e9 )
        #self.add_fps_line( ax, .5, 100*1e9 )
        fig.savefig( self.args.out_fn + "-ai" + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')

        # ai vs GF/s plot
    def set_bnds( self, ax, x, y ):
        self.x_min = min(x)
        self.x_max = max(x)*1.05
        self.y_min = min(y)
        self.y_max = max(y)*1.05
        ax.axis([self.x_min,self.x_max,self.y_min,self.y_max])

        self.data_aspect = float(self.x_max - self.x_min ) / (self.y_max - self.y_min)
        #self.axis_aspect = 0.618 * self.data_aspect
        #self.axis_aspect_rat = 1
        #self.axis_aspect = self.axis_aspect_rat * self.data_aspect
        #ax.set_aspect(self.axis_aspect)


    def add_fps_line( self, ax, tx, fps ):
        #Peak performance line and text
        x = [self.x_min,(self.x_min+self.x_max)/2,self.x_max]
        y = [ v/fps for v in x ]
        ax.plot(x,y, linewidth=1.0, color='black', linestyle=':' )

        label_string = "%.1fGF/s" % (fps/1e9)
        #yCoordinateTransformed = (log(peakPerf)-log(Y_MIN))/(log(Y_MAX/Y_MIN))
        print self.data_aspect
        ty = tx * self.data_aspect / fps
        #ty = .115
        rot=np.arctan(ty/tx) * 180 / np.pi
        ax.text(tx, ty, label_string, fontsize=8, transform=ax.transAxes, rotation=rot)



import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('--eff-fn', metavar="FN", type=str, default="eff-tab.raw", help="filename of eff values in latex table format" )
parser.add_argument('--out-fn', metavar="FN", type=str, default="eff", help="base filename of output plot image" )
parser.add_argument('--out-fmt', metavar="FN", type=str, default="png", help="extention/format for output plot image" )
args = parser.parse_args()
ep = EffPlot(args)

