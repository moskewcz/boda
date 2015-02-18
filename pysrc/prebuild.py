# Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
from nesi_gen import nesi_gen
import os

nesi_gen()

class DepFileProc( object ):
    def __init__( self ):
        for root, dirs, fns in os.walk( "." ):
            for fn in fns:
                if fn.endswith(".d"): self.proc_d_fn( fn )
            dirs[:] = []
    def proc_d_fn( self, fn ):
        lines = open(fn).readlines()
        #print lines

DepFileProc()
open( "prebuild.okay", "w" ).close()

