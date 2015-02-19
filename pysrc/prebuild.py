# Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
from nesi_gen import nesi_gen
from boda_make_util import DepFileProc, GenObjList

nesi_gen() # NESI c++ reflection system code generation
DepFileProc() # post-processing for gcc/makefile generated .d dependency files
GenObjList() # generate list of object file to build for make

# if we get here, we assume prebuild is done and good, and write an
# empty marker file that make can check for. the makefileremoves the
# file *before* running prebuild.py, and will abort if it is not
# present *after* running prebuild.py
open( "prebuild.okay", "w" ).close() 

