// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>
// Copyright (c) 2013, Forrest Iandola

//#include "mex.h"
#include<octave/oct.h>
#include<octave/octave.h>
#include<octave/parse.h>
#include<octave/mxarray.h>
#include<octave/mexproto.h>
//#include "CsCascade.h" //for read_timer
double read_timer( void ) { return 0.0d; }
#include "model.h"
#include<cassert>
// see: model.h for descriptions of each class field.

// handy accessors

// return field from struct a
static inline const mxArray *F(const mxArray *a, const char *field) {
    return mxGetField(a, 0, field);
}

// return pointer to field from struct a
template <typename T>
static inline T *Fpr(const mxArray *a, const char *field) {
    return mxGetPr(F(a, field));
}

// return scalar of field from struct a
template <typename T>
static inline T Fsc(const mxArray *a, const char *field) {
    return mxGetScalar(F(a, field));
}

// return field from struct in cell of struct array a
static inline const mxArray *CF(const mxArray *a, int cell, const char *field) {
    return F(mxGetCell(a, cell), field);
}

// for SSE/AVX vectorization
//assuming layout of (x*height*depth + y_pad*depth + d)
//@param dims[0,1,2] = height, width, depth
// typically, num_features == depth, but we explicitly pass in num_features to avoid confusion about PCA vs non-PCA.
int Model::get_y_padding(const int* dims, int num_features){
    int height= dims[0];
    int y_padding = height + num_features%32; //e.g. if num_features=8, pad by 8*(0,1,2, or 3) to get y-dim to be 32-aligned
    return y_padding;
}

// set up aligned memory, if needed. I think i'ts ok if num_features<NUM_FEATURES.
//@param dims[0,1] = height, width
//adapted from voc-release5/gdetect/fconvsse.cc
float* Model::copy_to_aligned(float *in, const int *dims, int num_features) {
    float *F = (float *)malloc_aligned(32, dims[0]*dims[1]*num_features*sizeof(float)); //align to 32-blocks (for hardware...not related to HOG dimensions)
    // Sanity check that memory is aligned
    if (!IS_ALIGNED(F)){
        printf("Memory not aligned -- in fconvsse_prepare()");
        exit(0);
    }

    for(int i=0; i<dims[0]*dims[1]*num_features; i++){
        F[i] = in[i];
    }
    return F;
}

// for SSE/AVX vectorization
// input layout: (x*height + y + d*height*width)
// output layout: (x*height*depth + y_pad*depth + d)
//@param dims[0,1,2] = height, width, depth
// typically, num_features == depth, but we explicitly pass in num_features to avoid confusion about PCA vs non-PCA. 
float* Model::transpose_and_align(float *in, const int *dims, int num_features) {
    int y_pad = get_y_padding(dims, num_features);

    assert(num_features == 8 || num_features == 32);

    float *F = (float *)malloc_aligned(32, y_pad*dims[1]*num_features*sizeof(float));
    // Sanity check that memory is aligned
    if (!IS_ALIGNED(F))
        mexErrMsgTxt("Memory not aligned");

    //TODO: OMP parallel for here. Need to do explicit pointer arithmetic.
    float *p = F;
    for (int x = 0; x < dims[1]; x++) {
        for (int y = 0; y < dims[0]; y++) {
            for (int f = 0; f < num_features; f++) {
                *(p++) = in[y + f*dims[0]*dims[1] + x*dims[0]];
            }
        }

        for (int y = dims[0]; y < y_pad ; y++) { //padding the end of a column, if necessary
            for (int f = 0; f < num_features; f++) {
                *(p++) = 0;
            }
        }
    }
    return F;
}

void Model::initmodel(const mxArray *model) {

    thresh        = Fsc<double>(model, "thresh");
    interval      = (int)Fsc<double>(model, "interval");
    numcomponents = (int)Fsc<double>(model, "numcomponents");
    sbin          = (int)Fsc<double>(model, "sbin");

    const mxArray *components = F(model, "components");
    const mxArray *definfos   = F(model, "defs");
    const mxArray *partinfos  = F(model, "partfilters");
    const mxArray *rootinfos  = F(model, "rootfilters");
    numpartfilters            = (int)(mxGetDimensions(partinfos)[1]);
    numdefparams              = (int)(mxGetDimensions(definfos)[1]);

    numparts        = new int[numcomponents];
    anchors         = new double**[numcomponents];
    defs            = new double*[numdefparams];
    rootfilters[0] = new float*[numcomponents]; //non-PCA
    rootfilters[1] = new float*[numcomponents]; //PCA

    partfilters[0]  = new float*[numpartfilters]; //non-PCA
    partfilters[1]  = new float*[numpartfilters]; //PCA
    rootfilterdims  = new mwSize*[numcomponents];
    partfilterdims  = new mwSize*[numpartfilters];
    pfind           = new int*[numcomponents];
    defind          = new int*[numcomponents];
    loc_scores      = new double*[numcomponents];

    for (int i = 0; i < numpartfilters; i++) {
        const mxArray *partinfo = mxGetCell(partinfos, i);

        const mxArray *w        = F(partinfo, "w");
        partfilterdims[i]       = (mwSize*)mxGetDimensions(w);
        //partfilters[0][i]       = (float *)mxGetPr(w);
        partfilters[0][i]       = transpose_and_align((float *)mxGetPr(w), partfilterdims[i], partfilterdims[i][2]); //numfeatures = partfilterdims[i][2] 

        w                       = F(partinfo, "wpca");
        mwSize *pcaSize = (mwSize*)mxGetDimensions(w); //should be == (partfilterdims[i][2] == pcadim)
        //partfilters[1][i]       = (float *)mxGetPr(w);
        partfilters[1][i]       = transpose_and_align((float *)mxGetPr(w), partfilterdims[i], pcaSize[2]);
    }
    for (int i = 0; i < numdefparams; i++) {
        const mxArray *definfo  = mxGetCell(definfos, i);
        defs[i]                 = Fpr<double>(definfo, "w");
    }
    const mxArray *cascadeinfo = F(model, "cascade");
    const mxArray *orderinfo   = F(cascadeinfo, "order");
    const mxArray *mxt         = F(cascadeinfo, "t");
    partorder                  = new int*[numcomponents];
    offsets                    = new double[numcomponents];
    t                          = new double*[numcomponents];

    for (int i = 0; i < numcomponents; i++) {
        const mxArray *parts  = CF(components, i, "parts");
        const mxArray *w      = CF(rootinfos, i, "w");
        numparts[i]           = mxGetDimensions(parts)[1];
        rootfilterdims[i]     = (mwSize*)mxGetDimensions(w); //we only have non-PCA root filters for now, since the PCA ones are handled in matlab -> fconvsse.cc
        //rootfilters[i]        = (float *)mxGetPr(w);
        rootfilters[0][i]        = transpose_and_align((float *)mxGetPr(w), rootfilterdims[i], rootfilterdims[i][2]);

        w      = CF(rootinfos, i, "wpca");
        mwSize *pcaSize = (mwSize*)mxGetDimensions(w); //should be == (rootfilterdims[i][2] == pcadim)
        rootfilters[1][i]        = transpose_and_align((float *)mxGetPr(w), rootfilterdims[i], pcaSize[2]); //PCA rootfilters

        anchors[i]            = new double*[numparts[i]];
        pfind[i]              = new int[numparts[i]];
        defind[i]             = new int[numparts[i]];
        offsets[i]            = mxGetScalar(mxGetField(mxGetCell(mxGetField(model, 0, "offsets"), i), 0, "w"));
        partorder[i]          = new int[2*numparts[i]+2];
        double *ord           = mxGetPr(mxGetCell(orderinfo, i));
        t[i]                  = mxGetPr(mxGetCell(mxt, i));

        for (int j = 0; j < numparts[i]; j++) {
            int dind                = (int)mxGetScalar(CF(parts, j, "defindex")) - 1;
            int pind                = (int)mxGetScalar(CF(parts, j, "partindex")) - 1;
            const mxArray *definfo  = mxGetCell(definfos, dind);
            anchors[i][j]           = Fpr<double>(definfo, "anchor");
            pfind[i][j]             = pind;
            defind[i][j]            = dind;
        }
        // subtract 1 so that non-root parts are zero-indexed
        for (int j = 0; j < 2*numparts[i]+2; j++)
            partorder[i][j] = (int)ord[j] - 1;
    }
}

void Model::initpyramid(const mxArray *pyramid, const mxArray *projpyramid) {
    mxArray *mx_feat = mxGetField(pyramid, 0, "feat");
    mxArray *mx_proj_feat = mxGetField(projpyramid, 0, "feat");
    numfeatures = mxGetDimensions(mxGetCell(mx_feat, 0))[2];
    pcadim = mxGetDimensions(mxGetCell(mx_proj_feat, 0))[2];

    numlevels    = (int)(mxGetDimensions(mx_feat)[0]);
    featdims     = new int*[numlevels];
    featdimsprod = new int[numlevels];
    feat[0]      = new float*[numlevels];
    feat[1]      = new float*[numlevels];

    double start_transpose = read_timer();

    #pragma omp parallel for schedule(static, 1) //chunk size = 1 -> round robin
    for (int l = 0; l < numlevels; l++) {

        // non-PCA pyramid
        const mxArray *mxA  = mxGetCell(mx_feat, l);
        featdims[l]         = (int*)mxGetDimensions(mxA);
        featdimsprod[l]     = featdims[l][0]*featdims[l][1];
        //feat[0][l]          = (float *)mxGetPr(mxA);
        feat[0][l]          = transpose_and_align((float *)mxGetPr(mxA), featdims[l], numfeatures);

        // projected pyramid
        mxA                 = mxGetCell(mx_proj_feat, l);
        //feat[1][l]          = (float *)mxGetPr(mxA);
        feat[1][l]          = transpose_and_align((float *)mxGetPr(mxA), featdims[l], pcadim);
    }
    double time_transpose = read_timer() - start_transpose;
    printf("        copied to aligned mem in %f sec \n", time_transpose);

    // Location/scale scores
    for (int c = 0; c < numcomponents; c++)
        loc_scores[c] = mxGetPr(mxGetCell(F(pyramid, "loc_scores"), c));
}

Model::~Model() {
  return;
    for (int i = 0; i < numcomponents; i++) {
        delete [] partorder[i];
        delete [] anchors[i];
        delete [] defind[i];
        delete [] pfind[i];
    }
    delete [] loc_scores;
    delete [] partorder;
    delete [] t;
    delete [] numparts;
    delete [] offsets;
    delete [] defind;
    delete [] pfind;
    delete [] anchors;
    delete [] defs;
    //TODO: free the *contents* of rootfilters and partfilters
    delete [] rootfilters[0];
    delete [] rootfilters[1];
    delete [] rootfilterdims;
    delete [] partfilters[0];
    delete [] partfilters[1];
    delete [] partfilterdims;
    delete [] featdims;
    delete [] featdimsprod;
    //TODO: free the *contents* of feat[0,1][:]
    delete [] feat[0];
    delete [] feat[1];
}
