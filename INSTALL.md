# WIP : Getting Started

## Boda config file setup (can be done after build if desired)
### note: all commands should be run from boda directory 

First, copy and rename my sample configuration file from the root directory to the lib dir as a template for yours:

    cp mwm_boda_cfg.xml lib/boda_cfg.xml

Then, edit the file. Important vars (see below): **alt_param_dir**, **pascal_data_dir**, **caffe_dir**

    emacs lib/boda_cfg.xml

Most variables are only used by specific modes, and in theory none are strictly required. However, due to limitations of the testing framework, currently all paths used by any tested mode must be specified -- but need not be valid. So, don't remove any variables, but don't worry about setting them correctly initially. 

For testing of CNN related code, you'll need binary caffe model parameter files matching (some of) the nets in the boda/nets directory. You can put the model files alongside the nets, or you can set **alt_param_dir** to an alternate location to look for the parameter files (see the comment in the config file itself).

Additionally, for running demos (perhaps those commands listed in the [doc/demo_notes.txt](doc/demo_notes.txt) file) **caffe_dir** should be a path to a checkout of caffe from which to find aux data files (not related to building). For example, the **cnet_predict** mode looks (by default) for the file "%(caffe_dir)/data/ilsvrc12/synset_words.txt".

Some tests may require various other inputs/data files/paths from the above config file to be set properly. Directly examination of the tests and their code may be necessary to determine the details, and some tests aren't going to be easy for anyone other than me to run without help / additional information. In particular, many tests use images (or a least a couple images) from the pascal VOC2007 dataset. So, you'll need to set **pascal_data_dir** to a copy of the VOC2007 data.

Here is some basic information on other vars:

bench_dir: related to hamming distance / obj. detection experiments. untested/unused? probably can ignore.

datasets_dir: some modes expect %(datasets_dir)/flickrlogos, %(datasets_dir)/imagenet_classification/ilsvrc12_val_lmdb, ...

flickr_data_dir: path to flickr logo data in VOC format, used by some older experiments only, untested, ignorable

dpm_fast_cascade_dir: path to dpm_fast_cascade codebase, for old DPM experiments, ignorable

ffld_dir: path to ffld, for old DPM experiments, ignorable

## Boda environment variables / bash completion setup (optional; can be done after build if desired)

The file [boda_env.sh](boda_env.sh) is designated as a place to specify useful environment variables for running boda, such as the location of the boda tree **BODA_HOME**. Note that this variable is unused outside boda_env.sh currently. The rest of the script will use BODA_HOME to put the **boda** binary in the PATH, as well as running a script to enable some limited, WIP support for bash completion for Boda. All of this is optional; if you prefer to manage PATH yourself, and/or if you don't desire bash completion support, there's no need to edit or run this script.

Finally, this file includes a line to add a directory to LD_LIBRARY_PATH so that the caffe shared library can be found. Note that this may or may not be needed depending on if boda is compiled with caffe support and how the caffe shared lib is built. Again this is optional -- if you have a different strategy for managing LD_LIBRARY_PATH, that's fine.

## Build

### Install needed dependencies

(some of the) needed extra packages under ubuntu 14.04 for a minimal compile:

    sudo apt-get install protobuf-compiler libprotobuf-dev liblmdb-dev libsparsehash-dev libpython-dev libboost-all-dev

some generally useful other packages for development:
    sudo apt-get install git emacs


### Build Configuration : editing [makefile](makefile) and [obj_list](obj_list)

    emacs obj_list

Generally, makefile need not be edited, as it's mostly a skelton used by the python-based pre-build system.
The file obj_list is where you control what software to link with and, if needed, any system specific paths. 
For example, to disable Caffe integration, change `[caffe]` to `[caffe disable]`. 
For initial builds, you may wish to disable various modules, in particular: octave, SDL2, caffe.
In general, for any non-disabled modules, modify the paths as needed. 
The build system will combine pieces of obj_list together with the Makefile template to form a complete Makefile.
This process is driven by regular make at the top level.

#### Notes on caffe support: short version: you probably want `[caffe disable]`

Enabling the `[caffe]` build stanza is *not* required for Boda to read caffe prototxt files.
Boda includes its own copy of the caffe protobuf message description file (`caffe.proto`) along with some small amount of copied and modified caffe protobuf related code (in `update_proto.cpp`), both in the boda/src/ext directory.
Using these files that are part of boda, 'basic' caffe support is controlled by the `[caffe_pb]` build stanza, which, unline the `[caffe]` stanza, probably should *not* be disabled, since many modes rely on it for reading input nets in caffe format.
Enabling `[caffe]` is only required if you desire using boda to use caffe to run nets *from inside boda* (a franken-middleware-mode, if you will), which is useful for certain forms of testing and profiling.
If you do enable `[caffe]` support, you must ensure that Boda's caffe.proto agrees with the one from the caffe build you choose.
I maintain a fork of caffe with a few patches applied to aid in integration with boda.
It should always be in a state that is 'good' for use with the current version of boda.

The main things that are currently patched are:
- tweaks to silence compiler warnings when including caffe headers (only included by caffe_fwd.cc)
- a hack to allow making dropout deterministic for testing

Neither are strictly required for enabling the `[caffe]` build stanza, though.
So, if you're not too picky about what caffe version to use, the easiest path to enable caffe integration for boda is to clone my fork of caffe, switch to the branch below, and point boda to it:

[https://github.com/moskewcz/caffe/tree/fix-sign-comp-in-hpps](https://github.com/moskewcz/caffe/tree/fix-sign-comp-in-hpps)

If you want to use your own caffe version, see boda issue #1 for the three options of how to proceed (TODO_DOCS: cleanup an inline here).

### compile

    make -j12

## Running Tests

### TODO: see boda issues for a starting place







