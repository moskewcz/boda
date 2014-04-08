// Copyright (c) 2013, Forrest Iandola
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h> //for uintptr_t

//macros from voc-release5 fconvsse.cc
#define IS_ALIGNED(ptr) ((((uintptr_t)(ptr)) & 0xF) == 0) 

#if !defined(__APPLE__)
#include <malloc.h>
#define malloc_aligned(a,b) memalign(a,b)
#else
#define malloc_aligned(a,b) malloc(b)
#endif

float* copy_to_aligned(float *in, const int *dims, int num_features_input);

class Model {
    public:
        // model data

        // size of HOG feature cell (e.g., 8 pixels)
        int sbin;
        // number of dimensions used for the PCA filter projection
        int pcadim;
        // dimenions of the HOG features
        int numfeatures;
        // component indexed array of root filters
        // rootfilters[0] holds non-PCA filters
        // rootfilters[1] holds PCA filters
        float **rootfilters[2];
        // sizes of root filters
        mwSize **rootfilterdims;
        // array of arrays of part filters
        // partfilters[0] holds non-PCA filters
        // partfilters[1] holds PCA filters
        float **partfilters[2];
        // dimensions of part filters
        mwSize **partfilterdims;
        // component indexed offset (a.k.a. bias) values 
        double *offsets;
        // location/scale scores
        double **loc_scores;
        // number of components in the model
        int numcomponents;
        // number of parts per component
        int *numparts;
        // global detection threshold
        double thresh;
        // component indexed arrays of part orderings
        int **partorder;
        // component indexed arrays of pruning thresholds
        double **t;
        // ideal relative positions for each deformation model
        double ***anchors;
        // array of deformation models
        double **defs;

        // maps from (component,part#) -> part filter or deformation model index
        // this enables supporting models with parts and deformation models that are
        // shared between components (not currently used)

        // map: pfind[component][part#] => part filter index
        int **pfind;
        // map: defind[component][part#] => def param index
        int **defind;

        // pooled part filter and def model counts
        int numpartfilters;
        int numdefparams;

        // feature pyramid data
        int numlevels;
        // dimensions of each feature pyramid level
        int **featdims;
        // number of positions in each feature pyramid level
        int *featdimsprod;
        // feature pyramid levels
        // feat[0] holds non-PCA HOG features
        // feat[1] holds PCA of HOG features
        float **feat[2];
        // number of levels per octave in feature pyramid
        int interval;

        // root PCA filter score + offset (stage 0 computed in cascade_detect.m)
        int numrootlocs;

        void initmodel(const mxArray *model);
        Model() {};
        Model(const mxArray *model) { initmodel(model); };
        ~Model();
        // fill in above model data using the mex struct pointed to by model
        // fill in above feature pyramid data using the mex structs pointed to
        // by pyramid and projpyramid
        void initpyramid(const mxArray *pyramid, const mxArray *projpyramid);

        //the following functions are for SSE / AVX vectorization
        float* copy_to_aligned(float *in, const int *dims, int num_features); 
        float* transpose_and_align(float *in, const int *dims, int num_features);
        // using static for get_y_padding so that CsCascade.cc can call get_y_padding without looking at the Model object.
        static int get_y_padding(const int* dims, int numfeatures); //round y-dim up to the nearest factor of 32
};

#endif /* MODEL_H */
