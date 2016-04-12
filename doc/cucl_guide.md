# CUCL guide

CUCL is boda's compatibility layer over OpenCL and CUDA.
Since OpenCL and CUDA are quite similar, it is possible to write lowest-common-denomination code that is compatible between the two.
For this to work, however, we must create an abstraction/interface above the level of both the OpenCL/CUDA languages and runtimes.
After that, there are additional (sort-of optional) layers of CUCL functionality related to code generation / metaprogramming.

CUCL is not intended to be a complete, unchanging thing.
Instead, it is intended to cover an existing set of use-cases with *reasonable* generality.
However, it is expected that new use-cases will require modifications at various layers.
Think of the CUCL support as simply part of the existing set of flows that has been factored out, not as anything so grandiose as a true language.

## CUCL runtime layer

The boda rtc (real-time-compilation) interface is [here](/src/rtc_compute.H>).

The OpenCL backend is [here](/src/ocl_util.cc).

The CUDA/nvrtc backend is [here](/src/nvrtc_util.cc).

Note that the backend files are the only files that use any OpenCL/CUDA headers/libraries/functionality. 
Also, they provide no direct interface -- they are strictly used via the rtc_compute_t interface class.

The rtc_compute_t interface provides for the compilation of code, management of variables/memory, and running functions from compiled code.

### CUCL runtime variables

Currently, all CUCL runtime variables are ND-Arrays of floats.
A variable has a string name, a dims_t (defined [here](/src/boda_base.H)) which defines its ND-Array dimensions, and backing memory on the compute device.
Currently, only un-padded, fixed-size, always-resident ND-Arrays are supported.
A dims_t primarily stores the number of dimensions of the ND-Array and the size of each dimension.
However, of particular note is that a dims_t can (optionally) store semantic names for dimension.
That is, instead of an ND-Array being 10x10x10, in boda an ND-Array can additional name its dimensions, such as Z=10,Y=10,X=10.
These names can be used by code to symbolically refer to and manipulate dimensions.
In particular, when checking that two dims_t are compatible (i.e the same 'type'), it is possible to check not only that the dimension sizes agree, but also that the names agree.

Note: a natural extension would be to support types other than float, perhaps by having a type field inside the dims_t.
Relaxing any of the other restrictions on usage would also be possible.

### Technical note
Of course, *some* other part(s) of the code must be aware that the backends exist in order to create them.
In boda, the NESI system provides the factory functionality to create instances of the backends behind the scenes
It relies on generated code with access to the backend code that is included into the backend .cc files.
This code/data is access elsewhere via usage of C function declarations and the like.

## CUCL language layer

The CUCL language layer consists of just a few #defines, listed here.

From the [OpenCL backend](/src/ocl_util.cc):


````
typedef unsigned uint32_t;
__constant uint32_t const U32_MAX = 0xffffffff;
typedef int int32_t;
//typedef long long int64_t;
#define CUCL_GLOBAL_KERNEL kernel
#define GASQ global
#define GLOB_ID_1D get_global_id(0)
#define LOC_ID_1D get_local_id(0)
#define GRP_ID_1D get_group_id(0)
#define LOC_SZ_1D get_local_size(0)
#define LOCSHAR_MEM local
#define LSMASQ local
#define BARRIER_SYNC barrier(CLK_LOCAL_MEM_FENCE)

// note: it seems OpenCL doesn't provide powf(), but instead overloads pow() for double and float. 
// so, we use this as a compatibility wrapper. 
// the casts should help uses that might expect implict casts from double->float when using powf() 
// ... or maybe that's a bad idea?
#define powf(v,e) pow((float)v,(float)e)
````

From the [CUDA/nvrtc backend](/src/nvrtc_util.cc):

````
typedef unsigned uint32_t;
uint32_t const U32_MAX = 0xffffffffU;
typedef int int32_t;
//typedef long long int64_t;
float const FLT_MAX = /*0x1.fffffep127f*/ 340282346638528859811704183484516925440.0f;
float const FLT_MIN = 1.175494350822287507969e-38f;
#define CUCL_GLOBAL_KERNEL extern "C" __global__
#define GASQ
#define GLOB_ID_1D (blockDim.x * blockIdx.x + threadIdx.x)
#define LOC_ID_1D (threadIdx.x)
#define GRP_ID_1D (blockIdx.x)
#define LOC_SZ_1D (blockDim.x)
#define LOCSHAR_MEM __shared__
#define LSMASQ
#define BARRIER_SYNC __syncthreads()
````

Again, this abstraction layer is not intended to be complete or unchanging, and may need to be extended over time.

## CUCL metaprogramming support

The above layers are sufficient to use CUCL via the rtc_compute_t interface.
However, to support metaprogramming and more automated management of function arguments, there is another layer of CUCL on top of the prior ones.
This lives [here](/src/rtc_func_gen.H>).
Rather than directly writing CUCL kernels, users of this layer instead write CUCL templates.
There are two basic phases by which CUCL templates are converted to CUCL functions:
- template parsing, which reads magic-comment-based CUCL annotations
- template instantiation, which performs string substitution on template parameters

### CUCL template parsing

This is phase in which all CUCL magic comments are read.
The relevant code is in the struct rtc_template_t in rtc_func_gen.H.

CUCL declarations (magic comments) all begin with "// CUCL", and then a space-seperated list of arguments.
The first argument is the type of declarations.
It is one of:

   if( (cd == "IN") || (cd == "INOUT") || (cd == "OUT") || (cd == "REF") || (cd == "IX") ) { ... }

With an optional suffix of "_MULTI" applied.

IN, INOUT, and OUT are used to declare function arguments, and must be immediately preceded by the name of the argument:

    GASQ float const * const in, // CUCL IN img:chan:y:x

REF is similar, and must also by immediately preceded by the name of the name of the reference:

/* work */  // CUCL REF pels_blk:out_chan_blk:pels_tile:out_chan_tile:pels:out_chan

In both cases, the CUCL declaration specified the names of the dimensions of the ND-Array for the bound name.
When a function is called, there must be a corresponding dims_t for each CUCL declaration, with a matching name.
Further, the names of the dimensions must also match between the declaration and the dims_t found in the calling environment.
Additionally, IN/INOUT/OUT declarations must correspond to a ND-Array in a CUCL variable.
REF declarations need not have data, but only just a dims_t -- they represent an ND-Array type, not an actually memory-backed ND-Array.
That is, REF declarations are a way to pass *just the dimensions* of an ND-Array to a template.

Finally, IX declarations are used to mark that particular a particular variable acts as an index for a particular set of ND-Array dimensions:

    // CUCL IX GLOB_ID_1D filts_ref

As opposed to the other declaration types, include index expression string as part of the declaration, and thus stand alone on their own line.
Later, we can see the usage of this index expression:

    val = filts_ref[GLOB_ID_1D];

Note that declaring GLOB_ID_1D, GRP_ID_1D, or LOC_ID_1D as indexed triggers special behaviour.
See rtc_call_gen_t init().
In particular, using these expressions as indexes will implicitly set the workgroup size and/or number of workgroups.
Note also that IX declarations can use only a subset of the dims_t that they refer to via the use_dims=... option.

### CUCL template variable synthesis

While some template variables are explicitly constructed in codegen or passed in, many are created from the various CUCL declarations.
See insert_nda_ix_exprs() and its calls.

### CUCL template instantiation

In this phase all, every string of the form %(template_var_name) in the CUCL template is replaced with the corresponding string value for the template variable 'template_var_name'.








