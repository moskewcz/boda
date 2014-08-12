// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"io_util.H"
#include"img_io.H"
#include"blf_pack.H"

namespace boda 
{
  using namespace boost;

  struct blf_bin_t {
    u32_box_t bin;
    vect_u32_box_t holes; 
    set_u32_box_t holes_set;
    // note: strict subset holes may be added in some situations, and thus we may later try to add
    // duplicate holes (which we drop). if we could drop/ignore subset holes easily, that might be
    // nice, but shouldn't be needed.
    void add_hole( u32_box_t const & hole ) { if( holes_set.insert( hole ).second ) { holes.push_back( hole ); } }
    blf_bin_t( u32_box_t const & bin_ ) : bin(bin_) { add_hole( bin ); }
    u32_pt_t place_box( u32_pt_t bsz ) {
      assert_st( bsz.both_dims_non_zero() );
      set_u32_box_t::const_iterator i = holes_set.begin();
      for( ; i != holes_set.end(); ++i ) { if( i->sz().both_dims_ge( bsz ) ) { break; } } // break if fits
      // if bsz didn't fit in any hole fit, return terminal/sentinal value
      if( i == holes_set.end() ) { return u32_pt_t_const_max; } 
      u32_box_t const ret( i->p[0], i->p[0] + bsz ); // place at -X,-Y in hole
      assert_st( ret.is_strictly_normalized() );
      // update holes
      vect_u32_box_t orig_holes;
      orig_holes.swap( holes );
      for( vect_u32_box_t::const_iterator i = orig_holes.begin(); i != orig_holes.end(); ++i ) {
	u32_box_t const ob = i->get_overlap_box_with( ret );
	if( ob.is_strictly_normalized() ) { // if the placement overlaps this hole
	  // remove hole from (ordered) set (note: implicitly removed from holes)
	  uint32_t const num_del = holes_set.erase( *i ); assert_st( num_del == 1 );
	  // add remaining/new holes from all sides if they exist (note that they may overlap each other)
	  for( uint32_t d = 0; d != 2; ++d ) { // +/- edge iter
	    for( uint32_t e = 0; e != 2; ++e ) { // dims (X/Y) edit iter
	      if( i->p[d].d[e] != ob.p[d].d[e] ) { // if there is a non-zero hole on this edge
		u32_box_t new_hole = *i; // start with original hole
		new_hole.p[d^1].d[e] = ob.p[d].d[e]; // trim to edge of overlap area
		assert_st( new_hole.is_strictly_normalized() );
		add_hole( new_hole ); 
	      }
	    }
	  }
	  //printf( "OVER (*i)=%s ret=%s\n", str((*i)).c_str(), str(ret).c_str() );
	} else { holes.push_back( *i ); } // no overlap with this hole, keep hole in holes (implicitly kept in holes_set)
      }
      //printf( "-- OUT holes=%s\n", str(holes).c_str() );
      return ret.p[0];
    }
  };

  typedef vector< blf_bin_t > vect_blf_bin_t; 
  typedef shared_ptr< blf_bin_t > p_blf_bin_t; 
  typedef vector< p_blf_bin_t > vect_p_blf_bin_t;

  uint32_t blf_place( vect_u32_pt_w_t & out, u32_box_t const & bin, vect_u32_pt_t const & to_place, bool const no_fit_okay )
  {
    vect_p_blf_bin_t bins;
    for( vect_u32_pt_t::const_iterator i = to_place.begin(); i != to_place.end(); ++i ) {
      u32_pt_t placement = u32_pt_t_const_max;
      uint32_t bin_ix = 0;
      for( ; bin_ix != bins.size(); ++bin_ix ) {
	placement = bins[bin_ix]->place_box( *i );
	if( placement != u32_pt_t_const_max ) { break; }
      }
      if( placement == u32_pt_t_const_max ) { // didn't fit in any bin, add one
	bins.push_back( p_blf_bin_t( new blf_bin_t( bin ) ) );
	placement = bins.back()->place_box( *i );
	if( placement == u32_pt_t_const_max ) {
	  if( no_fit_okay ) { bin_ix = uint32_t_const_max; } 
	  else {
	    rt_err( strprintf( "box (*i)=%s cannot be placed into empty bin of shape %s "
			       "(i.e. box to place > bin size)", str((*i)).c_str(), str(bin).c_str() ) );
	  }
	}
      }
      //printf( "placement=%s bin_ix=%s\n", str(placement).c_str(), str(bin_ix).c_str() );
      out.push_back( u32_pt_w_t( placement, bin_ix ) );
    }
    return bins.size();
  }


  struct blf_pack_t : virtual public nesi, public has_main_t // NESI(help="blf rectangle packing",
		      // bases=["has_main_t"], type_id="blf_pack")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="text output filename")
    filename_t to_pack_fn; //NESI(help="input: filename for list of boxes to pack",req=1)
    uint32_t bin_sz; //NESI(help="bin size for packing (as many bins needed will be used)",req=1)
    virtual void main( nesi_init_arg_t * nia ) { 
      p_ostream out = ofs_open( out_fn.exp );
      vect_u32_pt_t to_pack;
      read_text_file( to_pack, to_pack_fn.exp );
      sort( to_pack.begin(), to_pack.end(), u32_pt_t_by_prod_gt_t() );
      (*out) << strprintf( "bin_sz=%s\n", str(bin_sz).c_str() );
      (*out) << strprintf( "to_pack=%s\n", str(to_pack).c_str() );
      vect_u32_pt_w_t placements;
      blf_place( placements, u32_box_t( u32_pt_t(), u32_pt_t(bin_sz,bin_sz) ), to_pack, 0 );
      (*out) << strprintf( "placements=%s\n", str(placements).c_str() );
    }
  };


  // return a minimal amount of padding (4 per-edge padding values packed into a u32_box_t) for a
  // box of size sz, such that there is a minimum of min_pad[dim] on the -/+[dim] edges, and such
  // that both the - edge padding *and* the total of the padding + sz in each dim is a multiple of
  // align.
  u32_box_t pad_and_align_sz( u32_pt_t const & sz, u32_pt_t const & align, u32_pt_t const & min_pad ) {
    assert_st( sz.both_dims_non_zero() ); // might be okay, but doesn't seem sensible
    assert_st( align.both_dims_non_zero() );
    return u32_box_t( ceil_align( min_pad, align ), ceil_align( min_pad + sz, align ) - sz );
  }

  // create an approximately logarithmically spaced list of sizes given an input size, with
  // 'interval' steps per octave (factor of 2). the scale of the largest size returned will be
  // 2^num_upsampled_octaves. note that a 'primary' set of sizes (for the first downsampled octave)
  // are determined by scaling the input size by 2^(-i/interval) (with i in [0,interval) and
  // rounding to the nearest integer. the remaining sizes are calculated per-octave by iteratively
  // doubling the primary sizes for each upsampled octave, and by iteratively halving and rounding
  // the primary sizes for the downsampled octaves. this is done to allow for efficient and
  // close-to-right 2x up/downsampling to create the non-primary sizes, with the approximation being
  // the need to fudge 2x downsamples of odd sizes by (for example) clone-padding by 1 on the +
  // edge. sizes with either dimension == 1 won't be halved again, and thus the retuned set of sizes
  // is finite. due to rounding effects, one might want to re-calculate the scale of each primary
  // size by taking a ratio with the original size. it would also be possible to scale at exactly
  // the ideal scale and pad or clip the input or output slightly. for the down sampled sizes, the
  // best scale to use for them might depend on if exact-to-size versus exactly-2x scaling is
  // used. generally, as the intent is to use exactly-2x downsampling, the scales of the downsamples
  // sizes should be treated as exactly 2x less than thier parent size, even if the parent size was
  // odd. any scales smaller than and including the first duplicate size will be removed. roughly
  // log2(min_dim(in_sz))*interval sizes will be returned.
  void create_pyra_sizes( vect_u32_pt_t & pyra, u32_pt_t const & in_sz, 
			  uint32_t const num_upsamp_octaves, uint32_t const interval ) 
  {
    pyra.clear();
    assert_st( num_upsamp_octaves < 7 ); // sanity limit, maybe a little too strong?
    assert_st( in_sz.both_dims_non_zero() );
    assert_st( (in_sz.d[0] != uint32_t_const_max) && (in_sz.d[1] != uint32_t_const_max) ); // avoid overflow/sanity check
    for( uint32_t i = 0; i < interval; ++i ) { 
      double const scale = pow(2.0d, 0.0d - (double(i) / double(interval) ) );
      u32_pt_t scale_sz = in_sz.scale_and_round( scale );
      for( uint32_t oct_ix = 0; oct_ix != num_upsamp_octaves ; ++oct_ix ) { 
	u32_pt_t const us_scale_sz( scale_sz.d[0] << 1, scale_sz.d[1] << 1 ); // scale up one octave (factor of 2)
	assert_st( us_scale_sz.both_dims_gt( scale_sz ) ); // check for no overflow
	scale_sz = us_scale_sz;
      }      
      uint32_t cur_scale_ix = i;
      for( int32_t oct_ix = num_upsamp_octaves; oct_ix > -20 ; --oct_ix ) { // limit if -20 is another sanity-ish limit.
	assert_st( scale_sz.both_dims_non_zero() );
	while( pyra.size() <= cur_scale_ix ) { pyra.push_back( u32_pt_t() ); }
	pyra[cur_scale_ix] = scale_sz;
	if( (scale_sz.d[0] == 1) || (scale_sz.d[1] == 1) ) { break; } // more-or-less can't scale down more
	scale_sz = u32_pt_t( (scale_sz.d[0]+1) >> 1, (scale_sz.d[1]+1) >> 1 ); // scale down one octave (factor of 2)
	cur_scale_ix += interval;
      }
    }
    // remove all scales after and including first duplicate
    vect_u32_pt_t::iterator af = std::adjacent_find( pyra.begin(), pyra.end() );
    if( af != pyra.end() ) { pyra.erase( ++af, pyra.end() ); } 
  }

  void pyra_pack_t::do_place( conv_support_info_t const & csi ) {
    timer_t t("pyra_pack_do_place");
#if 0
    // note: we if the padding between images is 'neutral', we could
    // use ceil( support_sz / 2.0 ) padding on each image edge along
    // with some bin_edge_pad.
    u32_pt_t img_edge_pad = (csi.support_sz + u32_pt_t(1,1)) >> 1; // --> (img_edge_pad << 1) >= csi.support_sz
    // note: we might - csi.eff_tot_pad from bin_edge_pad here to
    // reduce wasted bin space. but. we choose not to because the
    // eff_tot_pad generally isn't equivalent to our internal padding.
    u32_box_t bin_edge_pad = {img_edge_pad,img_edge_pad};
#else
    // for now, we (conservatively) totally isolate each image. we don't need any bin_edge_pad in this case
    u32_pt_t img_edge_pad = csi.support_sz;
    u32_box_t bin_edge_pad;
#endif
    if( force_img_edge_pad ) { img_edge_pad = *force_img_edge_pad; }
    // increase - edge of bin padding so that (eff_tot_pad+bin_edge_pad) is a multiple of support_stride
    bin_edge_pad.p[0] = ceil_align( bin_edge_pad.p[0] + csi.eff_tot_pad.p[0], csi.support_stride ) - csi.eff_tot_pad.p[0];

    create_pyra_sizes( sizes, in_sz, num_upsamp_octaves, interval );
    vect_u32_pt_t to_pack;
    for( vect_u32_pt_t::const_iterator i = sizes.begin(); i != sizes.end(); ++i ) {
      pads.push_back( pad_and_align_sz( *i, csi.support_stride, img_edge_pad ) );
      to_pack.push_back( pads.back().bnds_sum() + (*i) );
    }
    // place the padded sizes (in to_pack) into bins of size bin_sz (with bin_edge_pad excluded from the edges)
    assert( bin_sz.both_dims_gt( bin_edge_pad.bnds_sum() ) ); // bin_edge_pad must leave some space left
    u32_pt_t const bin_sz_minus_pad = bin_sz - bin_edge_pad.bnds_sum();
    num_bins = blf_place( placements, u32_box_t( bin_edge_pad.p[0], bin_sz - bin_edge_pad.p[1] ), to_pack, 1 );

    // validate that results fit in bins (FIXME: add overlap check?)
    assert( sizes.size() == pads.size() );
    assert( sizes.size() == placements.size() );
    for( uint32_t i = 0; i != sizes.size(); ++i ) {
      if( placements[i].w == uint32_t_const_max ) { // didn't place, must be too big
	assert_st( !(sizes[i] + pads[i].bnds_sum()).both_dims_le( bin_sz_minus_pad ) ); // check was too big
      } else { // placed this size, check placement
	assert_st( sizes[i].both_dims_le( bin_sz ) );
	assert_st( placements[i].both_dims_lt( bin_sz ) );
	assert_st( pads[i].bnds_sum().both_dims_lt( bin_sz ) );
	assert_st( (placements[i]+pads[i].bnds_sum()+sizes[i]).both_dims_le( bin_sz ) );
      }
    }
    // adjust placements to be for the unpadded sizes
    for( uint32_t i = 0; i != sizes.size(); ++i ) { if( placements[i].w != uint32_t_const_max ) { placements[i] += pads[i].p[0];}}

  }

  void pyra_pack_cli_t::main( nesi_init_arg_t * nia ) { 
    if( !in_sz.both_dims_non_zero() ) { rt_err( "in_sz should be specified and non-zero in both dims" ); }
    p_ostream out = ofs_open( out_fn.exp );
    do_place( conv_support_info_t{min_pad,align,eff_tot_pad} );
    (*out) << strprintf( "sizes=%s\n", str(sizes).c_str() );
    (*out) << strprintf( "bin_sz=%s\n", str(bin_sz).c_str() );
    (*out) << strprintf( "num_bins=%s placements=%s\n", str(num_bins).c_str(), str(placements).c_str() );
  }

  void create_pyra_imgs( vect_p_img_t & pyra_imgs, p_img_t const & src_img, pyra_pack_t const & pyra ) {
    timer_t t("create_pyra_images");

    pyra_imgs.resize( pyra.sizes.size() );
    assert_st( get_wh(*src_img).both_dims_non_zero() );
    for( uint32_t i = 0; i < pyra.interval; ++i ) {
      if( i >= pyra.sizes.size() ) { break; } // < 1 octave in pyra, we're done
      u32_pt_t const i_max_sz = pyra.sizes[i];
      u32_pt_t const base_octave_sz = i_max_sz >> pyra.num_upsamp_octaves;
      if( ( base_octave_sz << pyra.num_upsamp_octaves ) != i_max_sz ) {
	rt_err( strprintf( "for interval step %s, i_max_sz=%s isn't evenly divisible by num_upsamp_octaves=%s.\n", 
			   str(i).c_str(), str(i_max_sz).c_str(), str(pyra.num_upsamp_octaves).c_str() ) );
      }
      p_img_t base_octave_img = src_img;
      if( !i ) { 
	if( base_octave_sz != get_wh(*src_img) ) {
	  rt_err( strprintf( "pyramid scale=1 size of base_octave_sz=%s does not equal src_img_sz=%s (pyra/src_img mismatch)\n", 
			     str(base_octave_sz).c_str(), str(get_wh(*src_img)).c_str() ) );
	} 
      } else {
	base_octave_img = downsample_to_size( src_img, base_octave_sz.d[0], base_octave_sz.d[1] );
      }
      p_img_t cur_octave_img = base_octave_img;
      // create base and (if any) upsampled octaves
      for( uint32_t j = i + (pyra.interval * pyra.num_upsamp_octaves); ; j -= pyra.interval ) {
	if( j < pyra_imgs.size() ) {
	  assert_st( pyra.sizes[j] == get_wh(*cur_octave_img) );
	  pyra_imgs[j] = cur_octave_img; 
	}
	if( j < pyra.interval ) { break; }
	cur_octave_img = upsample_2x( cur_octave_img );
      }
      cur_octave_img = base_octave_img;
      // create downsampled octaves
      for( uint32_t j = i + (pyra.interval * (1 + pyra.num_upsamp_octaves)); ; j += pyra.interval ) {
	if( ! (j < pyra_imgs.size()) ) { break; } // all done with ds'd octaves
	cur_octave_img = downsample_2x( cur_octave_img );
	assert_st( pyra.sizes[j] == get_wh(*cur_octave_img) );
	pyra_imgs[j] = cur_octave_img; 
      }
    }
  }


  void img_draw_box_pad( img_t * const dest, u32_box_t const & b, u32_box_t const & pad, uint32_t const & ec ) {
    timer_t t("img_draw_box_pad");
    // from the image edge to a distance of the min across all edge padding values, we interpolate
    // from the image border value to ec.
    uint32_t const min_edge_pad = std::min( pad.p[0].dims_min(), pad.p[1].dims_min() );
    if( !min_edge_pad ) { return; }

    // pad edges
    for( uint32_t e = 0; e != 2; ++e ) {
      for( uint32_t d = 0; d != 2; ++d ) {
	u32_pt_t sp;
	sp.d[d^1] = b.p[e].d[d^1]-e; // d coord of pixel on e edge
	for( sp.d[d] = b.p[0].d[d]; sp.d[d] != b.p[1].d[d]; ++sp.d[d] ) {
	  int32_t const stride = e?1:-1;
	  int32_t const stride_x = d?stride:0;
	  int32_t const stride_y = d?0:stride;
	  uint32_t const ic = dest->get_pel( sp.d[0], sp.d[1] );
	  img_draw_pels( dest, sp.d[0]+stride_x, sp.d[1]+stride_y, min_edge_pad/*pad.p[e].d[d^1]*/,
			 stride_x, stride_y, ic, ec ); 
	}
      }
    }
    // pad corners
	
    // dim 0 --> - side of image
    for( uint32_t dx = 0; dx != 2; ++ dx ) { 
      for( uint32_t dy = 0; dy != 2; ++ dy ) {
	u32_pt_t cp;
	cp.d[0] = b.p[dx].d[0]-dx; 
	cp.d[1] = b.p[dy].d[1]-dy;
	// cp is coord of dx,dy source (non-padding) corner image pixel 
	uint32_t const lt_sz = min_edge_pad; //std::min( pad.p[dx].d[0], pad.p[dy].d[1] ); 
	for( uint32_t dd = 2; dd <= lt_sz; ++dd ) {
	  uint32_t const cx = cp.d[0] + ( dx ? dd : -dd ); // cx,y is point on existing dx padding, dd outside image
	  uint32_t const cy = cp.d[1] + ( dy ? dd : -dd ); // x,cy is point on existing dy padding, dd outside image
	  uint32_t const cx_y_v = dest->get_pel( cx, cp.d[1] );
	  uint32_t const x_cy_v = dest->get_pel( cp.d[0], cy );
	  int32_t const stride_x = dx ? -1 :  1 ;
	  int32_t const stride_y = dy ?  1 : -1 ;
	  //printf( "cx=%s cp.d[1]=%s (dd-1)=%s stride_x=%s stride_y=%s\n", str(cx).c_str(), str(cp.d[1]).c_str(), str((dd-1)).c_str(), str(stride_x).c_str(), str(stride_y).c_str() );
	  img_draw_pels( dest, cx, cp.d[1], dd, stride_x, stride_y, cx_y_v, x_cy_v );  // note: overwrites cx,cp.d[1]
	}
      }
    }
  }

  void img_pyra_pack_cli_t::main( nesi_init_arg_t * nia ) { 
    if( !in_sz.is_zeros() ) { rt_err("for img_pyra_pack, don't set in_sz, it will be determined from the input image"); }
    timer_t t("img_pyra_pack_top");
    p_img_t img_in( new img_t );
    img_in->load_fn( img_in_fn.exp );
    in_sz.d[0] = img_in->w; in_sz.d[1] = img_in->h;
    do_place_imgs( conv_support_info_t{min_pad,align,eff_tot_pad}, img_in );
    if( write_images ) { 
      for( uint32_t bix = 0; bix != num_bins; ++bix ) {
	filename_t ofn = filename_t_printf( img_out_fn, str(bix).c_str() );
	printf( "ofn.exp=%s\n", str(ofn.exp).c_str() );	
	bin_imgs.at(bix)->save_fn_png( ofn.exp ); 
      }
    }
  }

  void img_pyra_pack_t::do_place_imgs( conv_support_info_t const & csi, p_img_t img_in ) {
    timer_t t("img_pyra_pack_do_place_imgs");
    do_place( csi );
    create_pyra_imgs( pyra_imgs, img_in, *this );
    uint32_t const inmc = 123U+(117U<<8)+(104U<<16)+(255U<<24); // RGBA

    {
      timer_t t2("img_pyra_pack_create_bins");
      for( uint32_t bix = 0; bix != num_bins; ++bix ) {
	bin_imgs.push_back( p_img_t( new img_t ) );
	bin_imgs.back()->set_sz_and_alloc_pels( bin_sz.d[0], bin_sz.d[1] ); // w, h
	bin_imgs.back()->fill_with_pel( inmc );
      }
    }
#pragma omp parallel for 
    for( uint32_t pix = 0; pix < pyra_imgs.size(); ++pix ) {
      //filename_t ofn = filename_t_printf( img_out_fn, str(pix).c_str() );
      //pyra_imgs[pix]->save_fn_png( ofn.exp );
      assert_st( get_wh(*pyra_imgs.at(pix)) == sizes.at(pix) );
      uint32_t const bix = placements.at(pix).w;
      if( bix == uint32_t_const_max ) { continue; } // skip failed placements FIXME: diagnostic?
      u32_pt_t const dest = placements.at(pix);
      img_copy_to( pyra_imgs.at(pix).get(), bin_imgs.at(bix).get(), dest.d[0], dest.d[1] );
      //printf( "dest=%s sizes.at(pix)=%s pads.at(pix)=%s\n", str(dest).c_str(), str(sizes.at(pix)).c_str(), str(pads.at(pix)).c_str() );
      img_draw_box_pad( bin_imgs.at(bix).get(), u32_box_t( dest, dest + sizes.at(pix) ), pads.at(pix), inmc );
    }

  }


// conversion functions for image<-->feature space
// handle scales / placements

  enum conv_mode_t : uint8_t { valid = 0, full = 1, core_valid = 2 };
  // valid: any returned output is computed using *only* values in the input value. that is, the
  // full support region of all returned outputs are fully contained in the input region.

  // full: the returned output consists of all outputs that using *any* of the input values (but may
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

  // note: for strides > 1, the 'full' and 'valid' cases are still sensible. however, the
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

  void in_box_to_out_box( i32_box_t & out_box, u32_box_t const & in_box, conv_mode_t const mode, 
			  conv_support_info_t const & csi ) 
  {    
    // calculate the full supporting in_box for the first out pixel (i.e. an out_box of (0,0) (1,1))
    u32_box_t const support( u32_pt_t(), csi.support_sz );
    // calculate the 'core' smaller supporting in_box for the same out pixel as above, such than the
    // cores are centered in the full support (i.e. are of size csi.support_stride) and tile to
    // cover in the input space with no gaps or overlap between the cores of adjacent output
    // pixels. note that the tiled cores may/will not extend to the borders of the input.
    u32_pt_t const support_core_lb = ( csi.support_sz + csi.support_stride + u32_pt_t( 1, 1 ) ) >> 1;
    u32_box_t const support_core( support_core_lb, support_core_lb + csi.support_stride );
    // choose which type support region we wish to use based on the mode
    u32_box_t const & need_valid_support = ( mode == core_valid ) ? support_core : support;
    assert( need_valid_support.is_strictly_normalized() );

    i32_box_t const in_pel = u32_to_i32(in_box + csi.eff_tot_pad.p[0]);
    assert_st( in_pel.is_strictly_normalized() );
    i32_box_t const out_valid = floor_div( in_pel - u32_to_i32( need_valid_support ), csi.support_stride );
    // check that out_valid is correct: it is valid, and minimal/maximal (one less/more in either dim is not valid)
    i32_box_t const in_used_pel = out_valid * u32_to_i32(csi.support_stride) + u32_to_i32( need_valid_support );
    assert_st( in_used_pel.is_strictly_normalized() );
    assert_st( in_pel.contains( in_used_pel ) ); // valid check
    assert_st( in_used_pel.p[0].both_dims_lt( in_pel.p[0] + u32_to_i32(csi.support_stride) ) ); // minimal check
    assert_st( in_used_pel.p[1].both_dims_gt( in_pel.p[1] - u32_to_i32(csi.support_stride) ) ); // maximal check

    // note1: out_box may not be strictly normalized. if it is not, it means the given input box
    // doesn't have any valid output features for the given mode.  note2: out_box may extend outside
    // the boundaries of the output space. in that case, it means that the given output pixels were
    // not calculated even though the input region covered the requested support (sub-)region. this
    // can only occur if the mode is 'core_valid' and if there is less than a certain amount of
    // padding/garbage in the input space. generally speaking, care must be taken when using
    // 'core_valid' mode to avoid using 'bad' parts of the input space.
    out_box = out_valid;
    out_box.p[1] += i32_pt_t(1,1);
  }



#include"gen/blf_pack.H.nesi_gen.cc"
#include"gen/blf_pack.cc.nesi_gen.cc"

};
