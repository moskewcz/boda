// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include<boost/algorithm/string.hpp>
#include"has_main.H"

namespace boda 
{
  using namespace boost;

  template< typename T >
  void read_text_file( vector< T > & out, string const & fn )
  {
    timer_t t("read_boxes_file");
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      T t;
      t.read_from_line_parts( parts, 0 );
      out.push_back( t );
    }
  }


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

#include"gen/blf_pack.cc.nesi_gen.cc"

};
