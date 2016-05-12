// Copyright (c) 2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _GBT_TILE_H_
#define _GBT_TILE_H_

#include"boda_base.H"
#include"geom_prim.H"

namespace boda 
{

  // returns td if it divides v 'well'; otherwise returns a 'good' divisor for v < td. this can be (and often is) == v.
  inline uint32_t get_good_div( uint32_t const & v, uint32_t const & td ) { // td == target divisor
    assert_st( v );
    if( v <= td ) { return v; } // if v is <= our target divisor, we will only have one chunk, and it need only be size v
    uint32_t d = td; // ideally, use our target divisor
    // assuming the work we do is u32_ceil_div(v,d)*d: while the waste is >= 20% of v, decrease d in order to decrease waste
    while( (u32_ceil_div(v,d)*d - v)*5 >= v ) { --d; } 
    assert_st( d );
    return d;
  }
  // tile an MxN space into three levels: MGxNG blocks of MBxNB threads, where each thread handles MTxNT elems. the
  // target number of M/N elements per thread (t_tile_sz) and total/max threads per block (max_tpb) are inputs.
  // note: max_tpb is treated as a target, but not to be exceeded
  struct gbt_tile_t {
    u32_pt_t num_blk; // total space in blocks; == ceil_div( num_thr, thr_per_blk )
    u32_pt_t thr_per_blk;
    u32_pt_t num_thr; // total space in threads; == ceil_div( num_mn, mn_per_thr )
    u32_pt_t mn_per_thr;
    u32_pt_t num_mn; // total space
    void init( u32_pt_t const & t_tile_sz, uint32_t const max_tpb, u32_pt_t const & num_mn_ ) {
      num_mn = num_mn_;
      //mn_per_thr.d[0] = t_tile_sz;
      mn_per_thr.d[0] = get_good_div( num_mn.d[0], t_tile_sz.d[0] );
      mn_per_thr.d[1] = t_tile_sz.d[1];
      mn_per_thr.d[1] = std::min( mn_per_thr.d[1], mn_per_thr.d[0]*3 ); // little benefit to higher aspect ratios
      //if( mn_per_thr.d[0] * 3 <= t_tile_sz ) { mn_per_thr.d[1] *= 2; } // experimental bump if small mn_per_thr.d[0]
      num_thr = ceil_div( num_mn, mn_per_thr );
        // set thr_per_blk.d[1] to sqrt( max_tpb ) min'd with num_thr.d[1] (no point in having more). also, min'd with
      // 3*num_thr.d[1] (other axis) to avoid higg (>3) block aspect ratios as they have little benefit (and some
      // downsides).
      thr_per_blk.d[1] = std::min( std::min( 16U, num_thr.d[0]*3 ), num_thr.d[1] );
      thr_per_blk.d[0] = 0;
      uint32_t best_tbp = 0;
      while( 1 ) {
	uint32_t const maybe_tbp = (thr_per_blk.d[0]+1) * thr_per_blk.d[1]; // recalculate proposed tpb
	if( maybe_tbp > max_tpb ) { break; }
	thr_per_blk.d[0] += 1;
	best_tbp = maybe_tbp;
	if( thr_per_blk.d[0] * mn_per_thr.d[0] >= num_mn.d[0] ) { break; } // no need to be larger
      }
      assert_st( best_tbp );
      assert_st( best_tbp <= max_tpb );
      assert_st( best_tbp == thr_per_blk.dims_prod() );
#if 0
      // now, try to increase thr_per_blk.d[1], in particular if we broke out due to m above
      while( 1 ) {
	uint32_t const maybe_tbp = thr_per_blk.d[0] * (thr_per_blk.d[1]+1); // recalculate proposed tpb
	if( maybe_tbp > max_tpb ) { break; }
	++thr_per_blk.d[1];
	best_tbp = maybe_tbp;
	if( thr_per_blk.d[1] * mn_per_thr.d[1] >= num_mn.d[1] ) { break; } // no need to be larger
      }
      assert_st( best_tbp );
      assert_st( best_tbp <= max_tpb );
#endif
      num_blk = ceil_div( num_thr, thr_per_blk );
    }
  };

}
#endif /* _GBT_TILE_H_ */
