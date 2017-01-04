# coding=UTF8


import numpy as np
from numpy import *
from pylab import *
import re
import random
import matplotlib
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
    float_str = "{0:.3g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def latex_float_with_math_env(f): return "$"+latex_float(f)+"$"

colors = [ "#5699D5", "#F6B26B", "#54873D" ] # light-ish blue, medium yellow, dark-ish green

vis = [ ]

        
class MiscPlot( object ):
    def __init__( self, args ):
        self.args = args
        self.do_plots()

    def do_plots( self ):

        #
        # python code to plot nonlinearity functions
        # written 2015 by Dan Stowell, dedicated to the public domain
        #plt.rcParams.update({'font.size': 12})


        evalpoints = np.linspace(-4, 4, 51)

        params = {
            'font.size': 12,
           'axes.labelsize': 12,
           'text.fontsize': 12,
           'legend.fontsize': 12,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
            'text.usetex': True,
            'figure.figsize': [7.5, 2.5],
           'lines.markersize': 4,
        }
        plt.rcParams.update(params)
        #plt.figure(frameon=False)
        #plt.axes(frameon=0)
        fig, (ax1,ax2) = plt.subplots(1,2)
        nlfuns = {
                'tanh':     (ax1,'b', '', lambda x: np.tanh(x)),
                'ReLU':     (ax2,'b', '', lambda x: np.maximum(0, x)),
        #        'Softplus':      ('g', '', lambda x: np.log(1 + np.exp( 1 * x))/ 1),
        #       'Softplus c=10': ('r', '', lambda x: np.log(1 + np.exp(10 * x))/10),
        #       'Exponential':   ('c', '', lambda x: np.exp(x)),
        }

        for nlname, (ax,color, marker, nlfun) in nlfuns.items():
                ax.set_title(nlname)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.axvline(0, color=[0.6]*3)
                ax.axhline(0, color=[0.6]*3)
                ax.plot(evalpoints, map(nlfun, evalpoints), label=nlname, color=color, marker=marker)
                ax.set_xlabel('x')
                ax.set_ylabel(nlname+'(x)')
                ax.xaxis.set_label_coords(0.5,-0.15)
                ax.yaxis.set_label_coords(-0.15,0.5)
        plt.tight_layout(w_pad=3)
        #plt.legend(loc='upper left', frameon=False)
        fig.savefig( self.args.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')
        #plt.savefig('tanh-and-relu.pdf')
        plt.close()
        return

        #background_color =(0.85,0.85,0.85) #'#C0C0C0'    
        #grid_color = 'white' #FAFAF7'
        background_color = "#ffffff"
        grid_color = 'black'
        rc('axes', facecolor = background_color)
        rc('axes', edgecolor = grid_color)
        #rc('axes', linewidth = 1.2)
        rc('axes', linewidth = 0.6)
        rc('axes', grid = True )
        rc('axes', axisbelow = True)
        rc('grid',color = grid_color)
        rc('grid',linestyle='-' )
        #rc('grid',linewidth=0.7 )
        rc('grid',linewidth=0.3 )
        #rc('xtick.major',size =0 )
        #rc('xtick.minor',size =0 )
        #rc('ytick.major',size =0 )
        #rc('ytick.minor',size =0 )




import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('--out-fn', metavar="FN", type=str, default="out", help="base filename of output plot image" )
parser.add_argument('--out-fmt', metavar="EXT", type=str, default="pdf", help="extention/format for output plot image" )
args = parser.parse_args()
ep = MiscPlot(args)
