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
import matplotlib.image as mpimg
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

from scipy.ndimage.filters import maximum_filter, convolve, correlate

class MiscPlot( object ):
    def __init__( self, args ):
        self.args = args
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
        for figname in args.fignames:
            self.out_fn = self.args.out_fn # may be none, if so, use default
            if self.out_fn is None: self.out_fn = figname
            ffn = "".join([ (ch if ch != "-" else "_") for ch in figname ])
            fig_func = getattr(self,ffn)
            fig_func()

    def tanh_and_relu( self ):
        #
        # python code to plot nonlinearity functions
        # written 2015 by Dan Stowell, dedicated to the public domain
        fig, (ax1,ax2) = plt.subplots(1,2)
        evalpoints = np.linspace(-4, 4, 51)
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
        fig.savefig( self.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')
        #plt.savefig('tanh-and-relu.pdf')
        plt.close()
        return

    def rgb_robot( self ):
        fig, (ax_rgb,ax_r,ax_g,ax_b) = plt.subplots(1,4)
        ax_rgb.set_title("RGB")
        img = mpimg.imread("../../test/robot.png")
        img = img[15:300:30,15:300:30,:]
        ax_sep = [ax_r,ax_g,ax_b]
        #print img
        ax_rgb.imshow(img,interpolation="nearest")
        for cix,ax in enumerate(ax_sep):
            img_sep = np.array(img)
            for c in range(img_sep.shape[2]):
                if c == cix: continue
                img_sep[:,:,c] = img_sep[:,:,cix] #np.zeros(img_sep.shape[:1]) 
            ax.set_title(("RED","GREEN","BLUE")[cix])
            ax.imshow(img_sep,interpolation="nearest")

        plt.tight_layout(w_pad=3)
        fig.savefig( self.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')
        plt.close()
        return

    def rgb_robot_pool( self ):
        fig, (ax,ax_mp,ax_ap) = plt.subplots(1,3)
        ax.set_title("RED")
        img = mpimg.imread("../../test/robot.png")
        img = img[15:300:30,15:300:30,0]
        #print img
        ax.imshow(img,cmap="gray",interpolation="nearest")
        img_mp = maximum_filter( img, footprint=np.ones((3,3)), mode="constant" )
        ax_mp.set_title("MAX-POOL")
        ax_mp.imshow(img_mp,cmap="gray",interpolation="nearest")
        img_ap = convolve( img, np.ones((3,3))/9, mode="constant" ) # FIXME: debatably not 'right' on borders ...
        ax_ap.set_title("AVG-POOL")
        ax_ap.imshow(img_ap,cmap="gray",interpolation="nearest")
        plt.tight_layout(w_pad=3)
        fig.savefig( self.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')
        plt.close()

    def show_conv( self, img, tag, ax, f ):
        oimg = correlate( img, f, mode="constant" ) # FIXME: debatably not 'right' on borders ...
        #print tag, oimg
        ax.set_title(tag)
        ax.imshow(oimg,cmap="gray",interpolation="nearest")
        

    def rgb_robot_conv( self ):
        fig, (ax,ax_ap,ax_xg,ax_yg) = plt.subplots(1,4)
        ax.set_title("RED")
        img = mpimg.imread("../../test/robot.png")
        img = img[15:300:30,15:300:30,0]
        #print img
        ax.imshow(img,cmap="gray",interpolation="nearest")
        zf = np.zeros((3,3))
        self.show_conv( img, "AVG-POOL", ax_ap, np.ones((3,3))/9 )
        xgf = np.copy(zf)
        xgf[(1,0)] = -1
        xgf[(1,2)] = 1
        #print xgf
        self.show_conv( img, "X-GRAD", ax_xg, xgf )
        ygf = np.copy(zf)
        ygf[(0,1)] = -1
        ygf[(2,1)] = 1
        self.show_conv( img, "Y-GRAD", ax_yg, ygf )
        plt.tight_layout(w_pad=3)
        fig.savefig( self.out_fn + "." + self.args.out_fmt, dpi=600,  bbox_inches='tight')
        plt.close()



import argparse
parser = argparse.ArgumentParser(description='Create eff plots.')
parser.add_argument('--out-fn', metavar="FN", type=str, help="override filename of output plot image" )
parser.add_argument('--out-fmt', metavar="EXT", type=str, default="pdf", help="extention/format for output plot image" )
parser.add_argument('fignames', metavar="FIGS", type=str, nargs="+", help="name(s) of figure(s) to create" )
args = parser.parse_args()
ep = MiscPlot(args)
