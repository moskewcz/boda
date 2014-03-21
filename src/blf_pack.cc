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

  void read_boxes_file( vect_u32_box_t & out, string const & fn )
  {
    timer_t t("read_boxes_file");
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      assert( parts.size() == 4 );
      u32_box_t b;
      b.p[0].d[0] = lc_str_u32( parts[0] );
      b.p[0].d[1] = lc_str_u32( parts[1] );
      b.p[1].d[0] = lc_str_u32( parts[2] );
      b.p[1].d[1] = lc_str_u32( parts[3] );
      out.push_back( b );
    }
  }


  struct blf_bin_t {
    u32_pt_t sz;
    vect_u32_box_t holes; 
    blf_bin_t( u32_pt_t const & sz_ ) : sz(sz_) { holes.push_back( u32_box_t( u32_pt_t(), sz ) ); }
    u32_pt_t place_box( u32_pt_t bsz ) {
      assert_st( bsz.both_dims_non_zero() );
      vect_u32_box_t::const_iterator i = holes.begin();
      for( ; i != holes.end(); ++i ) { if( i->sz().both_dims_ge( bsz ) ) { break; } } // break if fits
      if( i == holes.end() ) { return sz; } // didn't in any hole fit, return terminal/sentinal value (the sz of this bin)
      u32_box_t const ret( i->p[0], i->p[0] + bsz ); // place at -X,-Y in hole
      assert_st( ret.is_strictly_normalized() );
      // update holes
      vect_u32_box_t orig_holes;
      orig_holes.swap( holes );
      for( vect_u32_box_t::const_iterator i = orig_holes.begin(); i != orig_holes.end(); ++i ) {
	u32_box_t const ob = i->get_overlap_box_with( ret );
	if( ob.is_strictly_normalized() ) { // if the placement overlaps this hole
	  // add remaining/new holes from all sides if they exist (note that they may overlap each other)
	  for( uint32_t d = 0; d != 2; ++d ) { // +/- edge iter
	    for( uint32_t e = 0; e != 2; ++e ) { // dims (X/Y) edit iter
	      if( i->p[d].d[e] != ob.p[d].d[e] ) { // if there is a non-zero hole on this edge
		u32_box_t new_hole = *i; // start with original hole
		new_hole.p[d^1].d[e] = ob.p[d].d[e]; // trim to edge of overlap area
		assert_st( new_hole.is_strictly_normalized() );
		holes.push_back( new_hole ); 
	      }
	    }
	  }
	  printf( "OVER (*i)=%s ret=%s\n", str((*i)).c_str(), str(ret).c_str() );
	} else { holes.push_back( *i ); } // no overlap with this hole, keep unchanged
      }
      printf( "-- OUT holes=%s\n", str(holes).c_str() );
      return ret.p[0];
    }
  };

  struct blf_pack_t : virtual public nesi, public has_main_t // NESI(help="blf rectangle packing",bases=["has_main_t"], type_id="blf_pack")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t to_pack_fn; //NESI(help="input: filename for list of boxes to pack",req=1)
    filename_t pack_into_fn; //NESI(help="input: filename for list of boxes to pack",req=1)
    virtual void main( nesi_init_arg_t * nia ) { 
      vect_u32_box_t to_pack;
      read_boxes_file( to_pack, to_pack_fn.exp );
      vect_u32_box_t pack_into;
      read_boxes_file( pack_into, pack_into_fn.exp );
      printf( "to_pack=%s\npack_into=%s\n", str(to_pack).c_str(), str(pack_into).c_str() );

      if( pack_into.size() != 1 ) {
	rt_err( strprintf( "need exactly one box in pack_into, had %s", str( pack_into.size() ).c_str() ) );
      }
      
      blf_bin_t blf_bin( pack_into.front().sz() );
      for( vect_u32_box_t::const_iterator i = to_pack.begin(); i != to_pack.end(); ++i ) {
	blf_bin.place_box( i->sz() );
      }
    }
  };

#include"gen/blf_pack.cc.nesi_gen.cc"

};
