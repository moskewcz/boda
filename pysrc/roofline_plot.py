# original from: https://github.com/GeorgOfenbeck/perfplot (license unclear)

import sys
import os
import math
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

font = fm.FontProperties( family = 'Droid' )


background_color =(0.85,0.85,0.85) #'#C0C0C0'    
grid_color = 'white' #FAFAF7'
matplotlib.rc('axes', facecolor = background_color)
matplotlib.rc('axes', edgecolor = grid_color)
matplotlib.rc('axes', linewidth = 1.2)
matplotlib.rc('axes', grid = True )
matplotlib.rc('axes', axisbelow = True)
matplotlib.rc('grid',color = grid_color)
matplotlib.rc('grid',linestyle='-' )
matplotlib.rc('grid',linewidth=0.7 )
matplotlib.rc('xtick.major',size =0 )
matplotlib.rc('xtick.minor',size =0 )
matplotlib.rc('ytick.major',size =0 )
matplotlib.rc('ytick.minor',size =0 )
#matplotlib.rc('font', family='serif')


def knee_ai( perf, bw ): return perf / bw

def addPerfPt( alg, ai, perf ):
    ax.plot( [ai], [perf], alg.ls )

def addAILine( net_ai ):
    #Peak performance line and text
    y = np.linspace(Y_MIN, Y_MAX, 10)
    x = y*0.0 + net_ai.ai
    ax.plot( x, y, linewidth=0.75, color=net_ai.color, ls=net_ai.ls )

    #label_string = net_ai.get_lab_str()
    #xCoordinateTransformed = net_ai.ai # (log(ai)-log(X_MIN))/(log(X_MAX/X_MIN))
    #ax.text(xCoordinateTransformed+0.01, Y_MIN*1.1, label_string, fontsize=8, rotation=90, verticalalignment = 'bottom', horizontalalignment='right' )

def addPerfLine(peakPerf, label, kai, ls_ ):
    #Peak performance line and text
    x = np.linspace(kai, X_MAX, 10)
    y = x*0.0 + peakPerf
    ax.plot(x, y, linewidth=1.0, color='black', ls = ls_ )

    #ax.axhline(y=peakPerf, linewidth=0.75, color='black')
    label_string = label+" ("+str(peakPerf)+" GF/s)"
    yCoordinateTransformed = (log(peakPerf)-log(Y_MIN))/(log(Y_MAX/Y_MIN))
    ax.text(1 - len(label_string) / 110. - 0.01,yCoordinateTransformed+0.01, label_string, fontsize=8, transform=ax.transAxes)


def addBWLine(BW, label, kai, ls_ ):
    x = np.linspace(X_MIN, kai, 10)
    y = x*BW
    ax.plot(x, y, linewidth=1.0, color='black', ls = ls_ )
    yCoordinateTransformed = (log(X_MIN*BW)-log(Y_MIN))/(log(Y_MAX/Y_MIN))+0.16 #0.16 is the offset of the lower axis
    ax.text(X_MIN*1.1,(X_MIN*1.1*BW)*1.1, label+' ('+str(BW)+' GB/s)',fontsize=8, rotation=np.arctan(INVERSE_GOLDEN_RATIO * AXIS_ASPECT_RATIO) * 180 / np.pi, verticalalignment = 'bottom')
    #~ ax.text(0.01,yCoordinateTransformed+0.05+0.0075*(len(str(BW))-1), label+' ('+str(BW)+' B/C)',fontsize=8, rotation=45, transform=ax.transAxes)



X_MIN=0.1
X_MAX=1000.0
Y_MIN=1.0
Y_MAX=10000.0
#PEAK_PERF=8.0
#PEAK_BW=11.95
PEAK_PERF=[172.8, 5600.0 ]
PEAK_PERF_LABELS=['Adreno 418 Peak Compute','GTX-980 Peak Compute']
PEAK_BW=[15.0, 224.0 ]
PEAK_BW_LABELS = ['Adreno 418 Peak Bandwidth    (for reference only)','GTX-980 Peak Bandwidth']
knee_ais = [ knee_ai(a,b) for a,b in zip( PEAK_PERF, PEAK_BW ) ]
knee_ais = zip( knee_ais, ["--","-"] )

INVERSE_GOLDEN_RATIO=0.618
TITLE=""
X_LABEL="Arithmetic Intensity [Flops/Byte]"
Y_LABEL="Performance [GFlops/second]"
ANNOTATE_POINTS=1
AXIS_ASPECT_RATIO=log10(X_MAX/X_MIN)/log10(Y_MAX/Y_MIN)

class net_ai( object ):
    def __init__( self, color, ai, name, gfs ):
        self.ai = ai
        self.name = name
        self.gfs = gfs
        self.color = color
        self.ls = '-'
        if "-1-" in self.name: self.ls = '--'
    def get_lab_str( self ):
        return self.name +" ("+str(self.gfs)+" GF)" +" ("+str(self.ai)+" F/b)"
    def get_art( self ):
        return plt.Line2D((0,0),(1,0), color=self.color, marker='', linestyle=self.ls)

net_ais = dict( [ (t[2],net_ai(*t)) for t in [ 
    ('blue',8.95,"alexnet-1-image",2.27),
    ('blue',135,"alexnet-20-images",2.27*20),
    ('green',51.3,"nin-1-image",2.21),
    ('green',156.0,"nin-20-images",2.21*20),
    ('yellow',37.6,"googlenet-1-image",3.2),
    ('yellow',92.8,"googlenet-20-images",3.2*20),
    ('red',105,"firenet-v0-1-image",5.78),
    ('red',144,"firenet-v0-20-images",5.78*20),
    #    (1.1,".\hspace{6mm}stratos-1-image\hspace{5mm}",[13.6,6.0],0.286),(19.1,"stratos-20-images",[20.5,10.8],0.286*20),
    #    (0.7,"bigstride-1-image\hspace{4mm}",[8.0,3.6],0.097),(11.9,"bigstride-20-images",[12.5,6.8],0.097*20),
] ] )

class alg( object ):
    def __init__( self, name, ls, art, data ):
        self.name = name
        self.ls = ls
        self.art = art
        self.data = data
        self.perfs = dict( [ (name, net_ais[name].gfs * 1000.0 / ms) for (name,ms) in self.data.iteritems() ] )
        print self.name, self.perfs

cudnnv3_2015_05_caffe = alg( "cuDNNv3 (GTX 980)", "gs",  plt.Line2D((0,0),(0,0), color='g', marker='s', linestyle=''),
                             { # with 2015.05 version of caffe
                                 "alexnet-1-image":5.8,"alexnet-20-images":17.5,
                                 "nin-1-image":2.9,"nin-20-images":20.8,
                                 "googlenet-1-image":15.4,"googlenet-20-images":43.3,
                                "firenet-v0-1-image":4.3,"firenet-v0-20-images":49.5,
                             } )
boda_Q3 = alg("boda-rtc (CUDA *and* OCL) (GTX 980) Q3","gd", plt.Line2D((0,0),(0,0), color='g', marker='d', linestyle=''),
              { # exact version? 
                  "alexnet-1-image":14.9,"alexnet-20-images":28.8,
                  "nin-1-image":3.4,"nin-20-images":21.0,
                  "googlenet-1-image":18.3,"googlenet-20-images":59.3,
                  "firenet-v0-1-image":6.4,"firenet-v0-20-images":69.8,
              } )
boda_Q2 = alg("boda-nvrtc (CUDA) (GTX 980) Q2","go", plt.Line2D((0,0),(0,0), color='g', marker='o', linestyle=''),
              {
                  "alexnet-1-image":20.3,"alexnet-20-images":51.4,
                  "nin-1-image":10.5,"nin-20-images":43.3,
                  "googlenet-1-image":38.7,"googlenet-20-images":91.8,
                  #    "firenet-v0-1-image":0,"firenet-v0-20-images":0,
              } )
cudnnv2 = alg( "cuDNNv2 (GTX 980)", "g^",  plt.Line2D((0,0),(0,0), color='g', marker='^', linestyle=''),
               {
                   "alexnet-1-image":8.5,"alexnet-20-images":31.0,
                   "nin-1-image":5.1,"nin-20-images":27.3,
                   "googlenet-1-image":15.4,"googlenet-20-images":71.7,
                   #    "firenet-v0-1-image":0,"firenet-v0-20-images":0,
               } )

#algs = [ cudnnv3_2015_05_caffe, boda_Q3, boda_Q2, cudnnv2 ]
algs = [ cudnnv3_2015_05_caffe, boda_Q3 ]

fig = plt.figure()
# Returns the Axes instance
ax = fig.add_subplot(111)

#Log scale - Roofline is always log-log plot, so remove the condition if LOG_X
ax.set_yscale('log')
ax.set_xscale('log')

#formatting:
ax.set_title(TITLE,fontsize=14,fontweight='bold')
ax.set_xlabel(X_LABEL, fontproperties = font, fontsize=12)
ax.set_ylabel(Y_LABEL, fontproperties = font, fontsize=12)


#x-y range
ax.axis([X_MIN,X_MAX,Y_MIN,Y_MAX])
ax.set_aspect(INVERSE_GOLDEN_RATIO*AXIS_ASPECT_RATIO)

# Manually adjust xtick/ytick labels when log scale
locs, labels = xticks()
minloc =int(log10(X_MIN))
maxloc =int(log10(X_MAX) +1)
newlocs = []
newlabels = []
for i in range(minloc,maxloc):
    newlocs.append(10**i)
    # Do not plot the first label, it is ugly in the corner
    if i==minloc:
        newlabels.append('')
    elif i==maxloc-1: #Do not plot the last label either
        newlabels.append('')
    elif 10**i <= 100:
        newlabels.append(str(10**i))
    else:
        newlabels.append(r'$10^ %d$' %i)
xticks(newlocs, newlabels)

locs, labels = yticks()
minloc =int(log10(Y_MIN))
maxloc =int(log10(Y_MAX) +1)
newlocs = []
newlabels = []
for i in range(minloc,maxloc):
    newlocs.append(10**i)
    if i==minloc:
        newlabels.append('')
    elif 10**i <= 100:
        newlabels.append(str(10**i))
    else:
        newlabels.append(r'$10^ %d$' %i)
yticks(newlocs, newlabels)


for net_ai in net_ais.itervalues(): addAILine( net_ai )

print "KAIs",knee_ais
#Peak performance line and text
for p,l,kai in zip(PEAK_PERF, PEAK_PERF_LABELS, knee_ais): addPerfLine(p,l,kai[0], kai[1])
#BW line and text
for bw,l,kai in zip(PEAK_BW, PEAK_BW_LABELS, knee_ais ): addBWLine(bw,l,kai[0], kai[1])
#save file

out_fn = "cnn-gtx980-roofline%s.png"
fig.savefig( out_fn % "-no-perf", dpi=600,  bbox_inches='tight')

show_perf = 1

if show_perf:
    for alg in algs:
        for (net,perf) in alg.perfs.iteritems():
            addPerfPt( alg, net_ais[net].ai, perf )
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height])

    sorted_nets = [ (net.ai,net) for net in net_ais.itervalues() ]
    sorted_nets.sort()
    leg_art = [alg.art for alg in algs]
    leg_lab = [alg.name for alg in algs]
    leg_art += [net.get_art()  for (ai,net) in sorted_nets]
    leg_lab += [net.get_lab_str() for (ai,net) in sorted_nets]
    
    legend = ax.legend(leg_art,leg_lab,loc='upper center', shadow=True, fontsize='small',numpoints=1,ncol=2,bbox_to_anchor=[.5,1.35])
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#eeddcc')

fig.savefig( out_fn % "", dpi=600,  bbox_inches='tight')

#if not show_perf: show_perf_fn_part = "-no-perf"
# % (show_perf_fn_part,)


