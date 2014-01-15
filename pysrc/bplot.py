import matplotlib.pyplot as plt
import os.path

ospj = os.path.join

def plot_stuff( plt_fn, prc_elems, title ):
    plt.clf()
    plt.scatter(prc_elems[0], prc_elems[1] )
    plt.title( title )
    plt.savefig( plt_fn )
    return 0


def img_show( pels, save_as_filename ):
    plt.clf()
    plt.imshow( pels, interpolation="nearest" )
    if save_as_filename:
        plt.savefig( save_as_filename )
    else:
        plt.show()


def show_dets( pels, dets ):
    plt.clf()
    plt.imshow( pels, interpolation="nearest" )
    gca = plt.gca()
    #print dets
    for x1,y1,x2,y2 in dets:
        gca.add_patch(plt.Rectangle((x1,y1),x2-x1 ,y2-y1, ec="red",fill=0))
    plt.show()
