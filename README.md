## Boda: A C++ Framework for Efficient Experiments in Computer Vision [WIP]

Created by Matthew W. Moskewicz.

### License

Boda is BSD 2-Clause licensed; refer to the [LICENSE](LICENSE) file for the full License.

### Introduction

Boda is a one-grad-student, mostly-run-on-one-machine work in progress. 
However, for the brave, there is now a considerable amount of functionality present. 
Overall, documentation is still very much lacking. 
But, in particular, if you're interested in a unified C++ framework from camera/lmdbs/images to generated CUDA/OpenCL for CNNs, this might be an interesting project for you to explore. 
At this point, I think it's now plausible that others would be interested in and capable of usage of, experimentation with, and contributions to Boda. 
So, please file issues/PRs if that's the case.

### Getting Started / Installation

See the [INSTALL.md](INSTALL.md) file.

#### June 2016 paper
[Boda-RTC](https://arxiv.org/abs/1606.00094) (to appear WiMob 2016 NYC)

#### June 2016 poster
[Boda Poster Preview Slides](https://docs.google.com/presentation/d/1FFg-JZo2gUtnoqFl6HWWt0h6gMMFCD7ORFpXxPo5gl0/edit?usp=sharing)
[Boda Poster](https://drive.google.com/file/d/0B2T3gdjZVy_RVmU3MWJIZTRtalk/view?usp=sharing)

#### January 2016 poster
[Boda Poster Preview Slides](https://docs.google.com/presentation/d/170rZ7dDnMdc6VgTjfnZsWJcAuBr0x2m6RZxqkz08EfY/edit?usp=sharing)
[Boda Poster](https://drive.google.com/open?id=0B2T3gdjZVy_RVXRYNW9zbnA1eHM)

### Getting Started

There are 3 main config files for you to edit:

1. `cp mwm_boda_cfg.xml lib/boda_cfg.xml`. At the minimum, you should probably set: home_dir, caffe_dir, and models_dir. Others are mostly optional.
2. `obj_list`. This is where you control what software to link with. For example, to disable Caffe integration, change `[caffe]` to `[caffe disable]`. Modify the paths as needed.
3. `boda_env.sh` This is where you specify environment variables, such as the location of the `boda` binary and the locations of libs like `libcaffe.so`. This is optional -- if you have a different strategy for managing environment variables, that's fine too.

#### May 2014 Poster
[Boda Poster Preview Slides](https://docs.google.com/presentation/d/1kvyTOTBpmslKcxvPl4QF8NYlAbGriYA8IYOPL_dkSfw/edit?usp=sharing)
[Boda Poster](https://drive.google.com/file/d/0B2T3gdjZVy_RT1N6SkVoNFp1SmM/edit?usp=sharing)

#### Mid 2013 Poster
[Boda Poster Preview Slides](https://docs.google.com/presentation/d/15oa9wiLmeq5IsIo5wGjDm9_nMrw_aP4bc9pamKSoMd0/pub?start=false&loop=false&delayms=300000)
[Boda Poster](https://drive.google.com/file/d/0B2T3gdjZVy_RMXJ6MkprRlgyWUFXOGJBel8weFdZOWo2VFVn/edit?usp=sharing)

#### Boda modes/UI:

![Boda Overall Diagram](https://docs.google.com/drawings/d/1oir3fZt-SiO17C-vjsboLAkwucx4n6Le4kdqr_uEXFw/pub?w=670&h=266)

In the above diagram, the middle box is a Boda 'mode' -- a c++ class
with a main() function and a set of parameters. this is the boda
version of a standard c++ program with a C main() and some command
line argument processing such as gflags/getopt/etc. Boda makes it easy
to support many modes in a single binary/program, and provides some
magic comments / meta-programming to ease the burden of: 1) command
line / XML based UI creation for many such modes (with hierarchical
sharing of UIs / parameters) 2) testing (including automated
regression diffs over outputs) 3) timing / profiling.

The main 'magic' is a NEsted Structure Initialization system (NESI),
which uses magic comments, python, code generation, and a steaming
pile of down-home-style void pointers and C (or at least C style)
functions to initialize c++ structures from nested key/value trees (in
turn created from command line arguments and/or xml files), a la JSON
or the like.

