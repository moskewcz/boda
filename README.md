## Boda: A C++ Framework for Efficient Experiments in Computer Vision [WIP]

Created by Matthew W. Moskewicz.

### License

Boda is BSD 2-Clause licensed; refer to the [LICENSE](LICENSE) file for the full License.

### Introduction

Firstly, to be clear: Boda is a planned work / work in progress at an
early stage, and is probably not suitable for anyone to use for
anything at this point in time.

#### Mid 2013 Poster
[Boda Poster Preview Slides](https://docs.google.com/presentation/d/15oa9wiLmeq5IsIo5wGjDm9_nMrw_aP4bc9pamKSoMd0/pub?start=false&loop=false&delayms=300000)
[Boda Poster](https://drive.google.com/file/d/0B2T3gdjZVy_RMXJ6MkprRlgyWUFXOGJBel8weFdZOWo2VFVn/edit?usp=sharing)

#### A rough explanation of the Boda architechtural diagram in the poster

the middle box is a 'mode' -- a c++ class with a main() function and a
set of parameters. this is the boda version of a standard c++ program
with a C main() and some command line argument processing such as
gflags/getopt/etc. the pitch is that boda makes it easy to support
many modes in a single binary/program, and provides some magic
comments / meta-programming to ease the burden of:
1) command line / XML based UI creation for many such modes (with
hierarchical sharing of UIs / parameters)
2) testing (including automated regression diffs over outputs)
3) timing / profiling

the main 'magic' is a NEsted Structure Initialization system (NESI),
which uses magic comments, python, code generation, and a steaming
pile of down-home-style void pointers and C (or at least C style)
functions to initialize c++ structures from nested key/value trees (in
turn created from command line arguments and/or xml files), a la JSON
or the like.

