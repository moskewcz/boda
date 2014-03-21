/*
Copyright (C) 2013 Matthew W. Moskewicz
taken from voc-release5 features.cc and modified by  : Project webpage: http://www.cs.uchicago.edu/~rbg/latent/.
original copyright notice follows:


Copyright (C) 2011, 2012 Ross Girshick, Pedro Felzenszwalb
Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include"../boda_tu_base.H"
#include <math.h>
// small value, used to avoid division by zero
#define eps 0.0001

namespace boda {

// unit vectors used to compute gradient orientation
  double uu[9] = {1.0000, 
		  0.9397, 
		  0.7660, 
		  0.500, 
		  0.1736, 
		  -0.1736, 
		  -0.5000, 
		  -0.7660, 
		  -0.9397};
  double vv[9] = {0.0000, 
		  0.3420, 
		  0.6428, 
		  0.8660, 
		  0.9848, 
		  0.9848, 
		  0.8660, 
		  0.6428, 
		  0.3420};

  static inline float min(float x, float y) { return (x <= y ? x : y); }
  static inline float max(float x, float y) { return (x <= y ? y : x); }

  static inline int min(int x, int y) { return (x <= y ? x : y); }
  static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a double color image and a bin size 
// returns HOG features
  p_nda_double_t process( p_nda_double_t const & mximage, int const sbin ) {
    double *im = &mximage->cm_at1(0);

    dims_t const & dims_ = mximage->dims;
    vector< int > dims = {dims_.dims(2), dims_.dims(1), dims_.dims(0)};
    if( dims.size() != 3 || dims[2] != 3 ) {
      rt_err( "invalid input" );
    }

    // memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = (int)round((double)dims[0]/(double)sbin);
    blocks[1] = (int)round((double)dims[1]/(double)sbin);
    float *hist = (float *)calloc(blocks[0]*blocks[1]*18, sizeof(float));
    float *norm = (float *)calloc(blocks[0]*blocks[1], sizeof(float));

    // memory for HOG features
    dims_t out_dims;
    out_dims.resize_and_zero( 3 );
    out_dims.dims(0) = 27 + 4 + 1;
    out_dims.dims(1) = max(blocks[1]-2, 0);
    out_dims.dims(2) = max(blocks[0]-2, 0);
    p_nda_double_t mxfeat( new nda_double_t );
    mxfeat->set_dims( out_dims );
    double *feat = &mxfeat->cm_at1(0);

    vector< int > out = {out_dims.dims(2), out_dims.dims(1), out_dims.dims(0)};
  
    int visible[2];
    visible[0] = blocks[0]*sbin;
    visible[1] = blocks[1]*sbin;
  
    for (int x = 1; x < visible[1]-1; x++) {
      for (int y = 1; y < visible[0]-1; y++) {
	// first color channel
	double *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
	double dy = *(s+1) - *(s-1);
	double dx = *(s+dims[0]) - *(s-dims[0]);
	double v = dx*dx + dy*dy;

	// second color channel
	s += dims[0]*dims[1];
	double dy2 = *(s+1) - *(s-1);
	double dx2 = *(s+dims[0]) - *(s-dims[0]);
	double v2 = dx2*dx2 + dy2*dy2;

	// third color channel
	s += dims[0]*dims[1];
	double dy3 = *(s+1) - *(s-1);
	double dx3 = *(s+dims[0]) - *(s-dims[0]);
	double v3 = dx3*dx3 + dy3*dy3;

	// pick channel with strongest gradient
	if (v2 > v) {
	  v = v2;
	  dx = dx2;
	  dy = dy2;
	} 
	if (v3 > v) {
	  v = v3;
	  dx = dx3;
	  dy = dy3;
	}

	// snap to one of 18 orientations
	double best_dot = 0;
	int best_o = 0;
	for (int o = 0; o < 9; o++) {
	  double dot = uu[o]*dx + vv[o]*dy;
	  if (dot > best_dot) {
	    best_dot = dot;
	    best_o = o;
	  } else if (-dot > best_dot) {
	    best_dot = -dot;
	    best_o = o+9;
	  }
	}
      
	// add to 4 histograms around pixel using linear interpolation
	double xp = ((double)x+0.5)/(double)sbin - 0.5;
	double yp = ((double)y+0.5)/(double)sbin - 0.5;
	int ixp = (int)floor(xp);
	int iyp = (int)floor(yp);
	double vx0 = xp-ixp;
	double vy0 = yp-iyp;
	double vx1 = 1.0-vx0;
	double vy1 = 1.0-vy0;
	v = sqrt(v);

	if (ixp >= 0 && iyp >= 0) {
	  *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
	    vx1*vy1*v;
	}

	if (ixp+1 < blocks[1] && iyp >= 0) {
	  *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
	    vx0*vy1*v;
	}

	if (ixp >= 0 && iyp+1 < blocks[0]) {
	  *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
	    vx1*vy0*v;
	}

	if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
	  *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
	    vx0*vy0*v;
	}
      }
    }

    // compute energy in each block by summing over orientations
    for (int o = 0; o < 9; o++) {
      float *src1 = hist + o*blocks[0]*blocks[1];
      float *src2 = hist + (o+9)*blocks[0]*blocks[1];
      float *dst = norm;
      float *end = norm + blocks[1]*blocks[0];
      while (dst < end) {
	*(dst++) += (*src1 + *src2) * (*src1 + *src2);
	src1++;
	src2++;
      }
    }

    // compute features
    for (int x = 0; x < out[1]; x++) {
      for (int y = 0; y < out[0]; y++) {
	double *dst = feat + x*out[0] + y;      
	float *src, *p, n1, n2, n3, n4;

	p = norm + (x+1)*blocks[0] + y+1;
	n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
	p = norm + (x+1)*blocks[0] + y;
	n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
	p = norm + x*blocks[0] + y+1;
	n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
	p = norm + x*blocks[0] + y;      
	n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

	float t1 = 0;
	float t2 = 0;
	float t3 = 0;
	float t4 = 0;

	// contrast-sensitive features
	src = hist + (x+1)*blocks[0] + (y+1);
	for (int o = 0; o < 18; o++) {
	  float h1 = min(*src * n1, 0.2);
	  float h2 = min(*src * n2, 0.2);
	  float h3 = min(*src * n3, 0.2);
	  float h4 = min(*src * n4, 0.2);
	  *dst = 0.5 * (h1 + h2 + h3 + h4);
	  t1 += h1;
	  t2 += h2;
	  t3 += h3;
	  t4 += h4;
	  dst += out[0]*out[1];
	  src += blocks[0]*blocks[1];
	}

	// contrast-insensitive features
	src = hist + (x+1)*blocks[0] + (y+1);
	for (int o = 0; o < 9; o++) {
	  float sum = *src + *(src + 9*blocks[0]*blocks[1]);
	  float h1 = min(sum * n1, 0.2);
	  float h2 = min(sum * n2, 0.2);
	  float h3 = min(sum * n3, 0.2);
	  float h4 = min(sum * n4, 0.2);
	  *dst = 0.5 * (h1 + h2 + h3 + h4);
	  dst += out[0]*out[1];
	  src += blocks[0]*blocks[1];
	}

	// texture features
	*dst = 0.2357 * t1;
	dst += out[0]*out[1];
	*dst = 0.2357 * t2;
	dst += out[0]*out[1];
	*dst = 0.2357 * t3;
	dst += out[0]*out[1];
	*dst = 0.2357 * t4;

	// truncation feature
	dst += out[0]*out[1];
	*dst = 0;
      }
    }

    free(hist);
    free(norm);
    return mxfeat;
  }


}
