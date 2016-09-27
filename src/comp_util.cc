// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"

namespace boda {

  void comp_vars( std::ostream * const out, uint32_t & num_mad_fail,
		  double const mrd_toler, map_str_double const * const var_mrd_toler,
		  bool const diff_show_mrd_only, uint32_t const & max_err,
		  vect_string const & vns, p_map_str_p_nda_t const & vs1, p_map_str_p_nda_t const & vs2 ) 
  {
    (*out) << strprintf( "vars_to_compare: %s\n", str(vns).c_str() );
    for( vect_string::const_iterator i = vns.begin(); i != vns.end(); ++i ) {
      p_nda_t out_batch_1 = must_find( *vs1, *i );
      p_nda_t out_batch_2 = must_find( *vs2, *i );
      if( out_batch_1->elems_sz() != out_batch_2->elems_sz() ) {
	// treat size mismatch as warning, but print to log for regression tracking, since it may or may not be a 'real' error ...
	(*out) << strprintf( "%s: warning: size mismatch, can't compare: DIMS1[%s] DIMS2[%s]\n", 
			     i->c_str(), out_batch_1->dims.pretty_str().c_str(), out_batch_2->dims.pretty_str().c_str() );
	continue;
      }
      bool is_fail = 0;
      ssds_diff_t const ssds_diff(out_batch_1,out_batch_2);
      double vmt = var_mrd_toler ? get( *var_mrd_toler, *i, mrd_toler ) : mrd_toler;
      if( (ssds_diff.mrd >= vmt) || ssds_diff.has_nan() ) { ++num_mad_fail; is_fail = 1; }
      if( is_fail ) { // skip printing errors and details if no mad fail. set mrd_toler = 0 to force print (and failure)
	string diff_str;
	if( diff_show_mrd_only ) { diff_str = "MRD=" + str(ssds_diff.mrd); }
	else { diff_str = "ssds_str(out_batch_1,out_batch_2)=" + str(ssds_diff); }
	(*out) << strprintf( "%s: DIMS[%s] %s\n", i->c_str(), out_batch_1->dims.pretty_str().c_str(), diff_str.c_str() );
	uint32_t num_err = 0;
	assert_st( out_batch_1->dims == out_batch_2->dims );
	for( dims_iter_t di( out_batch_1->dims ) ; ; )  {
          double const dv = ssds_diff.pt->get_diff_double(di);
	  if( fabs(dv) >= vmt ) { 
            (*out) << ssds_diff.pt->get_diff_str(di); 
	    ++num_err;
	    if( num_err > max_err ) { break; }
	  }
	  if( !di.next() ) { break; }
	}
      }
    }
  }

  void comp_vars( std::ostream * const out, uint32_t & num_mad_fail,
		  double const mrd_toler, map_str_double const * const var_mrd_toler,
		  bool const diff_show_mrd_only, uint32_t const & max_err,
		  vect_string const & vns, p_map_str_p_nda_float_t const & vs1, p_map_str_p_nda_float_t const & vs2 ) 
  {
    comp_vars( out, num_mad_fail, mrd_toler, var_mrd_toler, diff_show_mrd_only, max_err, vns,
               upcast_map_str_p_nda_T_t( vs1 ), upcast_map_str_p_nda_T_t( vs2 ) );
  }


}
