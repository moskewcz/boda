import matplotlib.pyplot as plt

def plot_stuff( class_name, prc_elems ):
    print "plot",class_name
    plt.scatter(prc_elems[0], prc_elems[1] )
    plt.savefig(class_name+"_mAP"+".png")
    return 0
