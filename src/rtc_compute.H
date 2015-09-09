// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RTC_COMPUTE_H_
#define _RTC_COMPUTE_H_

#include"boda_base.H"

namespace boda 
{
  struct rtc_func_call_t;
  struct rtc_compute_t {
    virtual void init( void ) = 0;
    virtual void compile( string const & src, bool const show_compile_log, bool const enable_lineinfo ) = 0;

    // rp_float <-> var copies; var must exist; assumes the raw float pointer points to a block of floats of size at least as big as the var
    virtual void copy_to_var( string const & vn, float const * const v ) = 0;
    virtual void copy_from_var( float * const v, string const & vn ) = 0;
    virtual void create_var_with_sz_floats( string const & vn, uint32_t const & sz ) = 0;
    virtual uint32_t get_var_sz( string const & vn ) = 0;
    virtual void set_var_to_zero( string const & vn ) = 0;
    
    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    virtual void check_runnable( string const name, bool const show_func_attrs ) = 0;
    virtual void run( rtc_func_call_t & rfc ) = 0;
    virtual void finish_and_sync( void ) = 0;

    // non-virtual factored-out conversion functions layered on top of copy_{to,from}_var() and related functions
    // vect_float <-> var conversions/copies
    void init_var_from_vect_float( string const & vn, vect_float const & v );
    void set_vect_float_from_var( vect_float & v, string const & vn );
    void copy_var_to_nda( p_nda_float_t const & nda, string const & vn );
    void copy_nda_to_var( string const & vn, p_nda_float_t const & nda );

    p_nda_float_t copy_var_as_flat_nda( string const & vn );
    void copy_ndas_to_vars( vect_string const & names, map_str_p_nda_float_t const & ndas );
    void copy_vars_to_ndas( vect_string const & names, map_str_p_nda_float_t & ndas );

  };
  struct rtc_compute_t; typedef shared_ptr< rtc_compute_t > p_rtc_compute_t; 
}

#endif /* _RTC_COMPUTE_H_ */
