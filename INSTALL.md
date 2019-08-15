# WIP : Getting Started

## Install basic development packages

Before doing anything, you'll need a basic development setup. For Unbuntu 14.04/16.04, a good starting set of packages might be:

    sudo apt-get install git emacs vim build-essential libboost-all-dev

## Clone Boda Repo From Github

Next, you'll need to clone the repo, and decend into the fresh working copy:

    mkdir -p ~/git-work && cd ~/git-work && git clone http://github.com/moskewcz/boda && cd boda

## Build: installing boda dependencies, boda configuration, and boda compilation

### Install boda-specific dependencies

Depending on how it is configured, boda may require some additional dependencies.
Some of the less optional ones are listed below, and should be a good starting point.
All of these packages should be availible in Unbuntu 14.04/16.04:

    sudo apt-get install protobuf-compiler libprotobuf-dev liblmdb-dev libsparsehash-dev libjpeg-turbo8-dev 

Additionally, some additional optional dependencies are listed below.
Most/all can be disabled, see the following section for instructions on editing obj/obj_list to enable/disable dependencies.

### Python support

Note that the basic python development packages should have already been pulled in by libboost-all-dev, so they are not listed here specifically as a dependency.
However, boda's python support requires numpy support, which is listed below.

    sudo apt-get install python-numpy

### FFMPEG support

For video encoding/decoding. These two will pull in a few more on ubuntu 16.04/18.04 that seem to be enough. However, FFMPEG has had some ... interesting API/package-naming/forking issues over time, so this (and/or the code itself) might need some updates/changes over time.

    sudo apt-get install libavformat-dev libswscale-dev

#### SDL2 support:

For boda's minimal GUI/video-output support, you'll need the SDL2 development packages.
If you don't want those features, they can be disabled (see below section on editing obj_list).
Otherwise, this should install the needed packages:

    sudo apt-get install libsdl2-dev libsdl2-ttf-dev

#### Octave support:

For now, you should probably just disable the Octave integration support.
But if you're interested, perhaps you could start with the following and see what happens during build if you try to enable octave support:

    sudo apt-get install liboctave-dev

#### CUDA support:

For targeting nVidia GPUs via CUDA/nvrtc, you must install CUDA.
A good starting place is [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone).
If you are not using nVidia GPUs and/or don't wish to install/use Boda's CUDA/nvrtc support, disable cuda support.
Note that you must manually disable any cuda-dependent features as well, such as the culibs, nvrtc and caffe features.
If you forget, the build process will complain to remind you.


#### OpenCL support:

If you're using nVidia GPUS, CUDA provides OpenCL headers and an OpenCL runtime (ICD) for nVidia hardware.
Similarly, if you're using AMD GPUs, the APP SDK provides OpenCL headers and an OpencL runtime (ICD) for AMD hardware.
Intel also provides an OpenCL runtime/ICD, but usage of it has not been tested.
Additionally, Ubuntu provides packages with 'stock' versions of the OpenCL headers, and these should be usable for building boda as well.
Ubuntu also provides an open-source ICD loader (which is the libOpenCL that one actually links against), which should have been installed as part of the base system. 
Beyond just being able to build, the details of which flavors of OpenCL might actually work well are outside the scope of this document, but in general:
(1) OpenCL does not provide performance portability, and often not even correctness portability. 
(2) Targeting each specific hardware platform takes significant effort -- although a goal of boda is to easy this effort.
(3) Boda is currently aimed more toward 'discrete-GPU-like' devices, so attempting to add support for targeting CPUs via OpenCL would not be expected to be easy.

To install the Ubuntu 'stock' OpenCL headers and required-for-dev-symlink to the ubuntu/open-source OpenCL ICD loader library (which itself should already be installed): 

    sudo apt-get install opencl-headers ocl-icd-opencl-dev

However, note that while this will enable compilation of OpenCL support, it won't enable any OpenCL runtimes/ICDs, so things may not go well at runtime.
For that, install an ICD from NVidia, AMD, Intel, or whomever as discussed above.
In any event, OpenCL support may be disabled.

#### Caffe support:

In short, you probably want to disable caffe support.
However, if you do want caffe support, you'll need to build caffe.
See the longer note below for details.

### Build Configuration : editing [obj/makefile](obj/makefile) and [obj/obj_list](obj/obj_list)

    emacs obj/obj_list

Generally, the makefile need not be edited, as it's mostly a skeleton used by the python-based pre-build system.
The file obj_list is where you control what software to link with and, if needed, any system specific paths. 
For example, to disable Caffe integration, change `[caffe]` to `[caffe disable]`. 
For initial builds, you may wish to disable various modules as discussed above, in particular: octave, SDL2, caffe.
In general, for any non-disabled modules, modify the paths as needed.
The build system will combine pieces of obj_list together with the Makefile template to form a complete Makefile.
This process is driven by regular make at the top level. 
The makefile invokes the python build system by running [pysrc/prebuild.py](pysrc/prebuild.py)
The python part of the build system that processes obj_list lives in [pysrc/boda_make_util.py](pysrc/boda_make_util.py)
Note: Additionally, the python part of the build system that handles C++ metaprogramming/NESI code generation lives in [pysrc/nesi_gen.py](pysrc/nesi_gen.py). 
More documentation of this is TODO, but it should not require any special configuration or attention.

Each dependency of boda has a **build stanza** that starts with a header that minimally has the name of the dependency in []'s, such as `[cuda]` or `[SDL2]`. Each stanza is implicitly ended by the start of the next stanza. Various options can also be included in the in the header for each dependency:

    [foo needs=bar needs=baz] # this defines a dependency foo that has two sub-dependencies `bar` and `baz`
    [magiwrapper gen_fn=foo.cc] # uncommon (used only for protobuf related features currently); declares a build-time generated file
    [SDL2 disable] # used to disable a dependency; its build stanza will be ignored

The first stanza must be named `[base]`, and is special: it is always enabled.
It contains top-level boda dependencies and compiler setup.

If a dependency is not explicitly disabled, it will be enabled if it is used by any object file that is to be compiled or any other enabled dependency (i.e. recursively via sub-dependencies). 
However, if a dependency is explicitly disabled with the `disable` option, it will never be enabled. 
Any object files that depend on an explictly-disabled dependency will not be built.
Currently, if a disabled dependency is needed by an enabled dependcy, an error will be raised.
Note that it is unclear if this is a feature or limitation of the build system; the other natural behavior would be to recursively disable any parent dependencies of an explicitly-disabled dependency (and thus recursively disable any dependent object files).
If a dependency is enabled, then the contents of its build stanza are added to the makefile.
In particular, each stanza should add to LDFLAGS and CPPFLAGS to add any needed compiler/linker options to find needed header files and to find and link against any needed libraries.
For libraries not expected to be in the dynamic linker search path at runtime, a stanza may optionally use mechanisms like `rpath` as an alternative to using LD_LIBRARY_PATH at runtime.
By default, both the `caffe` and `cuda` dependencies use the rpath mechanism.
The default obj_list has reasonable default paths and options for all dependencies, but these must be manually altered as needed to match the local build environment if they are not correct for a given setup.

The final stanza must be named `[objs]`, and is special: it lists all the object files in boda along with what dependencies each has.
Files with no dependencies will always be compiled/linked.
Files that have dependencies will only be compiled/linked if all of their dependencies are enabled.
Generally, files that are not compiled/linked will only cause boda to be missing the features/modes/functionality provided by those files, but will not prevent linking.
In some case, a 'stub' version of a file/feature is provided for use when the dependency is unavailable; usually such stubs simply do nothing and/or emit an error if the needed functionality is used.
For example:

    [objs]
    ...
    img_io.o  # has no dependencies; will always be compiled and linked
    img_io-turbojpeg.o turbojpeg # depends on `turbojpeg`; if turbojpeg is disabled, this file will not be compiled/linked
    img_io-no-turbojpeg-stub.o -turbojpeg # anti-depends on turbojpeg; if turbojpeg is *not* disabled, this file will not be compiled/linked
    ...

If the user attempts to load a jpeg file using a build of boda where the turbojpeg dependency was disabled, a run-time error will be generated. 

Similar, for the python dependency we have: 

    [objs]
    ...
    pyif.o python
    pyif-no-python-stub.o -python
    ...

In general, usage of stubs allows for boda to unconditionally call some function. In this case, the boda python init function (py_init()) is always called from main(). This avoids the usage of #ifdef'd code depending on if python support is enabled. Instead, by linking either the real or stub version of the object, and thus of the py_init() function, we use the linker to manage optional feature dependencies.

The final result of obj_list processing will be written to the file `obj/dependencies.make`. This file is included in the root makefile, and can be inspected like any regular makefile.

#### Notes on caffe support: short version: you probably want `[caffe disable]`

Enabling the `[caffe]` build stanza is *not* required for Boda to read caffe prototxt files.
Boda includes its own copy of the caffe protobuf message description file (`caffe.proto`) along with some small amount of copied and modified caffe protobuf related code (in `update_proto.cpp`), both in the boda/src/ext directory.
Using just these files that are included in the boda distribution, 'basic' caffe support is controlled by the `[caffe_pb]` build stanza, which, unlike the `[caffe]` stanza, probably should *not* be disabled, since many modes rely on it for reading input nets in caffe protobuf format.
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

If you want to use your own caffe version, see boda issue #1 for the three options of how to proceed (TODO_DOCS: cleanup and inline here).

Note that, currently, boda assumes caffe is compiled with CUDA support, and thus if you don't have CUDA, you can't enable caffe.

### Compile

    make -C obj -j12 

or equivalently, if you prefer:

    cd obj && make -j12

## Boda config file setup (can be done before build if desired)

First, copy and rename the sample configuration file from the root directory to the lib dir as a template for yours:

    cp lib/boda_cfg.xml.example lib/boda_cfg.xml

Then, edit the file. Important vars (see below): **alt_param_dir**, **pascal_data_dir**, **caffe_dir**

    emacs lib/boda_cfg.xml

Most variables are only used by specific modes, and in theory none are strictly required.
However, due to limitations of the testing framework, currently all paths used by any tested mode must be specified -- but need not be valid.
So, don't remove any variables, but don't worry about setting them correctly initially. 

For testing of CNN related code, you'll need binary caffe model parameter files matching (some of) the nets in the boda/nets directory.
You can put the model files alongside the nets, or you can set **alt_param_dir** to an alternate location to look for the parameter files (see the comment in the config file itself).
The default **alt_param_dir** is set to %(boda_dir)/models (i.e. inside the WC/checkout).
However, this directory does not exist in the boda repo, due to the size of the model files (~1GB).
Boda reads a limited subset of standard caffe nets.
In particular, boda does not fully support FC/InnerProduct layers currently, since they are just a special case of Convolution.
Thus, the models that boda uses are generally just FC versions of existing common, stock models, where InnerProduct layers have been converted to Convolution layers using the boda cnet_fc_to_conv mode.
However, since the conversion process itself is undertested, undocumented, and manual, it is desierable to somehow provide 'ready to run' boda-flavor models as needed by the tests.
Currently, an experimental utility test-models-b2-util.py is provided that can download the needed models from a web server.
By default, the utiltiy will attempt to download the models from a b2 cloud storage bucket, but this bucket has a very limited download allowance (~1 full DL/day), so YMMV.

Some tests may require various other inputs/data files/paths from the above config file to be set properly.
Directly examination of the tests and their code may be necessary to determine the details, and some tests aren't going to be easy for anyone other than me to run without help / additional information.
In particular, many tests use images (or a least a couple images) from the pascal VOC2007 dataset.
So, you'll need to set **pascal_data_dir** to a copy of the VOC2007 data.
The default **pascal_data_dir** is set to %(boda_dir)/bench/VOCdevkit/VOC2007 (i.e. inside the WC/checkout).
The directory %(boda_dir)/bench/VOCdevkit is a gitlink/submodule pointing to a sibling repo of boda named ../VOCdevkit.
On github, in the moskewcz account, this repo exists as a sibling of boda, and contains a suitable subset of the VOC2007 data for running the boda tests.
Thus, doing the usual git submodule setup will give you the needed VOC2007 files for testing:

    git submodule init && git submodule update

Additionally, for running demos (perhaps those commands listed in the [doc/demo_notes.txt](doc/demo_notes.txt) file) **caffe_dir** should be a path to a checkout of caffe from which to find aux data files (not related to building). 
For example, the **cnet_predict** mode looks (by default) for the file "%(caffe_dir)/data/ilsvrc12/synset_words.txt".

Here is some basic information on other vars:

bench_dir: related to hamming distance / obj. detection experiments. untested/unused? probably can ignore.

datasets_dir: some modes expect %(datasets_dir)/flickrlogos, %(datasets_dir)/imagenet_classification/ilsvrc12_val_lmdb, ...

flickr_data_dir: path to flickr logo data in VOC format, used by some older experiments only, untested, ignorable

dpm_fast_cascade_dir: path to dpm_fast_cascade codebase, for old DPM experiments, ignorable

ffld_dir: path to ffld, for old DPM experiments, ignorable

## Boda environment setup: putting boda in the PATH, enabling bash completion support (optional; can be done before build if desired)

The file [scripts/boda_env.bash](scripts/boda_env.bash) is designated as a place to perform environment setup for running boda.
In particular, it puts the boda **lib** directory in the PATH (so that the **lib/boda** binary can be found in the PATH) as well as running a script to enable some limited, WIP support for bash completion for Boda. 
All of this is optional; if you prefer to manage PATH yourself, and/or if you don't desire bash completion support, there's no need to run this script.
In any event it should not need to be modified, and can be run from any location, once per shell, to modify the PATH and enable bash completion support.
To source it from the root of the boda tree:

    source scripts/boda_env.bash

## Development Notes

### Creating a TAGS file/table

The script [scripts/gen_etags.sh](scripts/gen_etags.sh) can be sourced (or run) from the root of the boda tree to produce a emacs TAGS file:

    ./scripts/gen_etags.sh

## Testing

### Getting Needed Model Files and Datasets

See the above section on boda config file setup for information on getting ready-to-run versions of the models and the subset of the PASCAL VOC2007 dataset used by the tests.
But, in short, use test-models-b2-util.py to download models, and use the provided git submodule for the needed subset of teh PASCAL VOC2007 dataset.

### Running Tests

From the boda/run directory, make a test-running directory, and cd into it.
From there, run the test_all mode.
Depending on which features you have enabled/disabled, you should see something like the below.
Note that many of the tests do require caffe support, and so if you've disabled that, many individual tests will fail to initialize, since the 'caffe' computation backend won't exist.
So, the part above about not needing caffe enabled above doesn't currently really hold true if you want to run the tests.


````
moskewcz@GPU86AA:~/git_work/boda/run/tr1$ time boda test_all ; date
WARNING: test_cmds: some modes had test commands that failed to initialize. perhaps these modes aren't enabled?
  FAILING MODES: oct_featpyra oct_resize run_dfc test_oct
TIMERS:  CNT     TOT_DUR      AVG_DUR    TAG  
           4     165.071s      41.267s    test_all_subtest
          44     165.055s       3.751s    test_cmds_cmd
          39      7.050ms      0.180ms    diff_command
          31    294.271ms      9.492ms    read_pascal_image_list_file
           2      7.246ms      3.623ms    read_results_file
           1      0.491ms      0.491ms    score_results_for_class
           4      0.076ms      0.019ms    read_text_file
          18      71.860s       3.992s    nvrtc_compile
        3843     89.661ms      0.023ms    cu_launch_and_sync
        1288    627.221ms      0.486ms    caffe_copy_layer_blob_data
           1    139.204ms    139.204ms    caffe_init
          48      42.978s    895.390ms    caffe_create_net
         948       1.361s      1.436ms    caffe_set_layer_blob_data
         749     82.954ms      0.110ms    img_copy_to
         894       2.531s      2.831ms    subtract_mean_and_copy_img_to_batch
          20    146.598ms      7.329ms    dense_cnn
         689    173.205ms      0.251ms    caffe_fwd_t::set_vars
         689       2.982s      4.328ms    caffe_fwd_t::run_fwd
         689       2.132s      3.095ms    caffe_fwd_t::get_vars
        3190       2.107s      0.660ms    caffe_copy_output_blob_data
         588       2.656s      4.518ms    sparse_cnn
          60       2.458s     40.979ms    net_upsamp_cnn
          60    279.103ms      4.651ms    upsample_2x
          60    816.201ms     13.603ms    img_upsamp_cnn
          81     60.967ms      0.752ms    conv_pipe_fwd_t::set_vars
          81       4.436s     54.776ms    conv_pipe_fwd_t::run_fwd
          81       2.817s     34.782ms    conv_pipe_fwd_t::get_vars
        1392       1.081s      0.777ms    caffe_get_layer_blob_data

real	1m43.818s
user	1m50.968s
sys	0m5.536s
Fri Mar  4 18:01:15 PST 2016
````