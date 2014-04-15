// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"io_util.H"

namespace boda 
{
  using namespace boost;

  struct blf_bin_t {
    u32_box_t bin;
    vect_u32_box_t holes; 
    set_u32_box_t holes_set;
    void add_hole( u32_box_t const & hole ) {
      holes.push_back( hole );
      bool const did_ins = holes_set.insert( hole ).second; assert_st( did_ins );
    }
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

  void blf_place( vect_u32_pt_w_t & out, u32_pt_t bin_sz, vect_u32_pt_t const & to_place )
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
	bins.push_back( p_blf_bin_t( new blf_bin_t( u32_box_t( u32_pt_t(), bin_sz ) ) ) );
	placement = bins.back()->place_box( *i );
	if( placement == u32_pt_t_const_max ) {
	  rt_err( strprintf( "box (*i)=%s cannot be placed into empty bin with bin_sz=%s "
			     "(i.e. box to place > bin size)", str((*i)).c_str(), str(bin_sz).c_str() ) );
	}
      }
      //printf( "placement=%s bin_ix=%s\n", str(placement).c_str(), str(bin_ix).c_str() );
      out.push_back( u32_pt_w_t( placement, bin_ix ) );
    }
  }


  struct blf_pack_t : virtual public nesi, public has_main_t // NESI(help="blf rectangle packing",bases=["has_main_t"], type_id="blf_pack")
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
      blf_place( placements, u32_pt_t(bin_sz,bin_sz), to_pack );
      (*out) << strprintf( "placements=%s\n", str(placements).c_str() );
    }
  };

  u32_pt_t pad_and_align_sz( u32_pt_t const & sz, u32_pt_t const & align ) {
    assert_st( sz.both_dims_non_zero() ); // might be okay, but doesn't seem sensible
    assert_st( align.both_dims_non_zero() );
    assert(0); // TODO
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

  struct pyra_pack_t : virtual public nesi, public has_main_t // NESI(help="pyramid packing",bases=["has_main_t"], type_id="pyra_pack")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    u32_pt_t in_sz; //NESI(help="input size to create pyramid for",req=1)
    uint32_t num_upsamp_octaves; //NESI(default=1,help="number of upsampled octaves")
    uint32_t interval; //NESI(default=10,help="steps per octave (factor of 2)")
    virtual void main( nesi_init_arg_t * nia ) { 
      p_ostream out = p_ostream( &std::cout, null_deleter<std::ostream>() ); //ofs_open( out_fn.exp );
      vect_u32_pt_t pyra;
      create_pyra_sizes( pyra, in_sz, num_upsamp_octaves, interval );
      (*out) << strprintf( "pyra=%s\n", str(pyra).c_str() );
    }
  };

#include"gen/blf_pack.cc.nesi_gen.cc"

};
