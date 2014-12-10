// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"conv_common.H"
#include"str_util.H"
namespace boda 
{

// conversion functions for image<-->feature space
// handle scales / placements

  // valid: any returned output is computed using *only* values in the input value. that is, the
  // full support region of all returned outputs are fully contained in the input region.

  // any_valid: the returned output consists of all outputs that using *any* of the input values (but may
  // also use padding or even garbage when the padding is anti-convervative )

  // core_valid: like 'valid', but only requires that a central core region of each returned
  // output's support region be contained in the input region. thus, it returns a larger area than
  // the 'valid' mode. the size of the core region is equal to the support stride of the output,
  // such that the core regions of the output tile to cover the input without overlaps or gaps. 

  // note: in core_valid mode, if there is an odd/even mismatch between the support stride and the
  // support size (per dim) then the boundaries of the core region will not align to the input
  // grid. in this case, the core is shifted by +.5 pixels (per dim) to align to the input grid.

  // note: in the case where the stride=1, and the support size is odd, and there is sufficient
  // effective total support padding, then 'core_valid' mode yields an output area equal to the
  // input area, like the 'same' mode of matlab's 2D convolution. if the support size is even, the
  // shifting of the core as above should match the arbitrary choice of matlab's 'same' mode to
  // still return an output of the same size as the input by favoring keeping - outputs (due to the
  // +.5 shifting of the core) when clipping the central region.

  // note: for strides > 1, the 'any_valid' and 'valid' cases are still sensible. however, the
  // 'core_valid' case is a bit more problematic. the returned size will be approximately the input
  // size / the stride, and hopefully the overall behavior is still sensible, but the semantics of
  // the boundary/edge cases are a bit unclear/aribitrary (but are (intended to be) exactly as
  // described above).

  // notes on padding: (1) for the 'valid' mode, the amount of padding will not effect the returned
  // size of out_box or even the set of descriptors that it contains. however, since the padding
  // may/will create additional outputs on the - edge of the output space, out_box will shift in the
  // + direction for a given input window as padding increases. (2) for the 'core_valid' mode, there
  // may be descriptors that need to use padding. thus, a lack of sufficient padding will cause
  // descriptors that should be associated with some valid core support region in the input to not
  // exist.

  // we define the 'base' output support as the support of the single 'first' output pixel from
  // (0,0),(1,1). based on core_only, we return either the entire support or only the 'core' support
  // as per the above definition. the cores are centered in the full support (i.e. are of size
  // csi.support_stride) and tile to cover in the input space with no gaps or overlap between the
  // cores of adjacent output pixels. note that the tiled cores may/will not extend to the borders
  // of the input.
  i32_box_t get_base_out_support( conv_support_info_t const & csi, bool const core_only ) { 
    assert( csi.support_sz.both_dims_non_zero() );
    if( !core_only ) { return i32_box_t( i32_pt_t(), u32_to_i32( csi.support_sz ) ); }
    else {
      i32_pt_t const support_core_ub = u32_to_i32( ( csi.support_sz + csi.support_stride + u32_pt_t( 1, 1 ) ) >> 1 );
      return i32_box_t( support_core_ub - u32_to_i32( csi.support_stride ), support_core_ub ); 
    }
  }
  
  void in_box_to_out_box( i32_box_t & out_box, u32_box_t const & in_box, conv_mode_t const mode, 
			  conv_support_info_t const & csi ) 
  { 
    // shift input region to be in an input space that included the total effective input padding
    i32_box_t const in_pel = u32_to_i32(in_box + csi.eff_tot_pad.p[0]);
    assert_st( in_pel.is_strictly_normalized() );
    // choose which type support region we wish to use based on the mode
    if( mode == cm_any_valid ) {
      i32_box_t const support = get_base_out_support( csi, 0 );
      out_box.p[0] = ceil_div(  in_pel.p[0] + i32_pt_t(1,1) - support.p[1], csi.support_stride );
      out_box.p[1] = floor_div( in_pel.p[1] - i32_pt_t(1,1) - support.p[0], csi.support_stride );
    } else if( mode == cm_valid || mode == cm_core_valid ) {
      i32_box_t const support = get_base_out_support( csi, ( mode == cm_core_valid ) );
      // note: zero area is okay. if denormalized, though, there will no output area (closed out_box
      // will be denormal and the final open out_box will not be strictly normal.
      i32_box_t const in_pel_shrunk = in_pel - support; 
      out_box = i32_box_t{ ceil_div( in_pel_shrunk.p[0], csi.support_stride ), 
			   floor_div( in_pel_shrunk.p[1], csi.support_stride ) }; // out_box is closed here
      // check that out_valid is correct: it is valid, and minimal/maximal (one less/more in either dim is not valid)
      i32_box_t const in_used_pel = support + ( out_box * u32_to_i32(csi.support_stride) );
      //printf( "in_pel=%s support=%s in_used_pel=%s out_box=%s\n", 
      //      str(in_pel).c_str(), str(support).c_str(), str(in_used_pel).c_str(), str(out_box).c_str() );
      // FIXME: the first (and other?) assert fails (as one might expect) if out_box is not
      // normalized ('negative'/no output area case). skip the asserts in that case?
      assert_st( in_used_pel.is_strictly_normalized() ); 
      assert_st( in_pel.contains( in_used_pel ) ); // valid check
      assert_st( in_used_pel.p[0].both_dims_lt( in_pel.p[0] + u32_to_i32(csi.support_stride) ) ); // minimal check
      assert_st( in_used_pel.p[1].both_dims_gt( in_pel.p[1] - u32_to_i32(csi.support_stride) ) ); // maximal check
    } else { assert(!"unknown mode"); }

    // convert the output space box from closed [] to half-open [) pixel coverage semantics.  after
    // the conversion, the box area (as returned by box.get_area()) == the covered pixels. that is,
    // the box corners/edges are 'between' pixels.
    out_box.p[1] += i32_pt_t(1,1);

    // note1: at this point, out_box may not be strictly normalized (even after the []->[) adj.). if
    // it is not, it means the given input box doesn't have any valid output features for the given
    // mode.  note2: out_box may extend outside the boundaries of the output space. in that case, it
    // means that the given output pixels were not calculated even though the input region covered
    // the requested support (sub-)region. this can only occur if the mode is 'core_valid' and if
    // there is less than a certain amount of padding/garbage in the input space. generally
    // speaking, care must be taken when using 'core_valid' mode to avoid using 'bad' parts of the
    // input space.
  }

  // return the input-space support of the given out_box (which must be strictly normalized). the
  // 'unchecked_' means that no clipping/checking of either out_box or in_box to any notion of a
  // 'valid' input or output space is performed by this function. also, note that there might be
  // multiple in_boxes that yield the same out_box due to non-unit strides. the details of which
  // sets of in_boxes can map to the same out_box and which in_box is returned by this function for
  // a out_box vary by mode; see the per-case comments.
  void unchecked_out_box_to_in_box( i32_box_t & in_box, i32_box_t const & out_box, conv_mode_t const mode, 
			  conv_support_info_t const & csi ) 
  { 
    assert_st( out_box.is_strictly_normalized() );
    i32_box_t out_box_closed( out_box.p[0], out_box.p[1] - i32_pt_t(1,1) );
    if( mode == cm_any_valid ) {
      // for the 'any_valid' mode, the returned in_box is minimal: any smaller box would map to a
      // smaller out_box. [TODO: CHECK: thus, the returned in_box box can be grown by up to
      // (stride-1) in any combination of edges and will still yield the same out_box. ] note: some
      // out_boxes are too small to map to any in_box in any_valid mode; i.e. they are smaller than
      // the out_box that would be returned from a single pixel of input space. we return a
      // denormalized in_box in this case, but it's unclear if that makes sense / is useful.
      i32_box_t const support = get_base_out_support( csi, 0 );
      // lb of in_box is -corner of +most (full) support pixel in -most output pixel
      in_box.p[0] = support.p[1] - i32_pt_t(1,1) + out_box_closed.p[0] * u32_to_i32(csi.support_stride);
      // ub of in_box is +corner of -most (full) support pixel in +most output pixel
      in_box.p[1] = support.p[0] + i32_pt_t(1,1) + out_box_closed.p[1] * u32_to_i32(csi.support_stride);
    } else if( mode == cm_valid || mode == cm_core_valid ) {
      // for 'valid' and 'core_valid' modes, the returned support is exactly the union of the
      // support of each pixel in the out_box, based on the above definitions of the supporting
      // region. thus it is the minimal in_box that maps to this out_box. [TODO: CHECK: if the
      // resultant in_box is expanded by up to (csi.support_stride-1) on any combination of edges,
      // it will yield the same out_box. thus, if the stride is 1, the retured in_box is a unique
      // mapping for this out_box. ]
      i32_box_t const support = get_base_out_support( csi, ( mode == cm_core_valid ) );
      in_box = support + ( out_box_closed * u32_to_i32(csi.support_stride) );
    } else { assert(!"unknown mode"); }
    in_box = in_box - u32_to_i32(csi.eff_tot_pad.p[0]); // adjust for padding
  }

  // this wrapper: (1) includes a special case to handle when global pooling is involved, and thus must take the
  // full input size so it can be returned in that case. (2) returns valid and core_valid boxes (in one call)
  void unchecked_out_box_to_in_boxes( i32_box_t & valid_in_box, i32_box_t & core_valid_in_box, 
				      i32_box_t const & out_box, conv_support_info_t const & csi,
				      u32_pt_t const & full_in_sz ) 
  { 
    if( !csi.support_sz.is_zeros() ) {
      unchecked_out_box_to_in_box( valid_in_box, out_box, cm_valid, csi );
      unchecked_out_box_to_in_box( core_valid_in_box, out_box, cm_core_valid, csi );
    } else {
      valid_in_box = core_valid_in_box = i32_box_t{{},u32_to_i32(full_in_sz)}; // whole image
    }
  }  

}
