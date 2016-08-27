# WIP : Getting Started

## Install basic development packages

Before doing anything, you'll need a basic development setup. For Unbuntu 14.04/16.04, a good starting set of packages might be:

    sudo apt-get install git emacs vim build-essential libboost-all-dev

## Clone Boda Repo From Github

Next, you'll need to clone the repo, and decend into the fresh working copy:

    mkdir -p ~/git-work && cd ~/git-work && git clone http://github.com/moskewcz/boda && cd boda

## Build

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

### compile

    make -C obj -j12 

or equivalently, if you prefer:

    cd obj && make -j12


## Development Notes

### Creating a TAGS file/table

The script [scripts/gen_etags.sh](scripts/gen_etags.sh) can be sourced (or run) from the root of the boda tree to produce a emacs TAGS file:

    ./scripts/gen_etags.sh

## Running Tests

### copying needed model files

Assuming you have all the needed model files on some machine in /scratch/models, using where each model is /scratch/models/$net/best.caffemodel, you can use rsync to copy them to another machine. The list of models needed for testing (hopefully up to date, but otherwise a good starting place) is in boda/test/test_all_nets.txt. Here is a sample command to copy the models from the local machine to a remote machine named 'targ_machine':

    # FIXME: UNTESTED!!! needs a /scratch dir already setup, maybe with +t / sticky-bit set like /tmp ...
    moskewcz@maaya:~/git_work/boda$ for net in `cat test/test_all_nets.txt`; do ssh targ_machine mkdir -p /scratch/models/$net ; scp {,targ_machine:}/scratch/models/$net/best.caffemodel; done

### copying needed dataset files

Also, the images from the VOC-2007 dataset are used by the tests. Again, assuming they are in /scratch/datasets/VOC-2007/VOCdevkit (with the actual images and such in /scratch/datasets/VOC-2007/VOCdevkit/VOC2007/{JpegImages,ImageSets,Annotations}):

    rsync -a /scratch/datasets/VOC-2007/VOCdevkit targ_machine:/scratch/datasets/VOC-2007

### running the tests

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