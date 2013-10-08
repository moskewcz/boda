import matplotlib.pyplot as plt
import os.path

ospj = os.path.join

def plot_stuff( class_name, prc_elems ):
    print "plot",class_name
    plt.clf()
    plt.scatter(prc_elems[0], prc_elems[1] )
    plt.savefig(class_name+"_mAP"+".png")
    return 0


def img_show( pels, save_as_filename ):
    plt.clf()
    plt.imshow( pels, interpolation="nearest" )
    if save_as_filename:
        plt.savefig( save_as_filename )
    else:
        plt.show()

