# Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
from nesi_gen import nesi_gen
from boda_make_util import DepFileProc, GenObjList
import argparse

parser = argparse.ArgumentParser(description='Boda prebuild/makefile-generation system.')
parser.add_argument('--obj-list-fn', metavar="FN", type=str, help="filename of objects/dependencies list", required=True )
args = parser.parse_args()

DepFileProc() # post-processing for gcc/makefile generated .d dependency files
gol = GenObjList( args.obj_list_fn ) # generate list of object file to build for make
nesi_gen( gol ) # NESI c++ reflection system code generation

# if we get here, we assume prebuild is done and good, and write an
# empty marker file that make can check for. the makefileremoves the
# file *before* running prebuild.py, and will abort if it is not
# present *after* running prebuild.py
open( "prebuild.okay", "w" ).close() 

