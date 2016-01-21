// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RTC_COMPUTE_H_
#define _RTC_COMPUTE_H_

#include"boda_base.H"

namespace boda 
{
  struct rtc_func_call_t;

  struct rtc_compute_t : virtual public nesi // NESI(help="modes that can be used for rtc",is_abstract=1,tid_vn="be")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string be; //NESI(help="name of back-end to use for rtc",req=1)

    virtual void init( void ) = 0;
    virtual void compile( string const & src, bool const show_compile_log, bool const enable_lineinfo ) = 0;

    // rp_float <-> var copies; var must exist; assumes the raw float pointer points to a block of floats of size at least as big as the var
    virtual void copy_to_var( string const & vn, float const * const v ) = 0;
    virtual void copy_from_var( float * const v, string const & vn ) = 0;
    virtual void create_var_with_dims_floats( string const & vn, dims_t const & dims ) = 0;
    virtual dims_t get_var_dims_floats( string const & vn ) = 0;
    uint32_t get_var_sz_floats( string const & vn );
    virtual void set_var_to_zero( string const & vn ) = 0;
    
    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    virtual void check_runnable( string const name, bool const show_func_attrs ) = 0;
    // note: run() overwrites / fills in the call_id field. it is valid after run() returns until realease_per_call_id_data() is called.
    virtual void run( rtc_func_call_t & rfc ) = 0; 
    virtual void finish_and_sync( void ) = 0;
    virtual void release_per_call_id_data( void ) = 0; // invalidates all call_ids inside rtc_func_call_t's

    // get duration from start of call b to end of call e (where b may == e to time a single call).

    // we almost-only need to get the timing of a single call, but we allow spanning calls for convenience and/or to allow
    // detecting inconsistencies/overheads between the sum of per-call times and the overall all-calls time (as proxied by
    // calling get_dur( first_call, last_call ) ).
    virtual float get_dur( uint32_t const & b, uint32_t const & e ) = 0; // note: uses call_id field, which must be valid

    virtual void profile_start( void ) = 0;
    virtual void profile_stop( void ) = 0;

    // non-virtual factored-out conversion functions layered on top of copy_{to,from}_var() and related functions
    // vect_float <-> var conversions/copies
    void init_var_from_vect_float( string const & vn, vect_float const & v );
    void set_vect_float_from_var( vect_float & v, string const & vn );
    void copy_var_to_nda( p_nda_float_t const & nda, string const & vn );
    void copy_nda_to_var( string const & vn, p_nda_float_t const & nda );

    p_nda_float_t copy_var_as_flat_nda( string const & vn );
    void copy_ndas_to_vars( vect_string const & names, map_str_p_nda_float_t const & ndas );
    void copy_vars_to_ndas( vect_string const & names, map_str_p_nda_float_t & ndas );

    void create_var_from_nda( p_nda_float_t const & nda, string const & vn );
    p_nda_float_t create_nda_from_var( string const & vn );

  };
  struct rtc_compute_t; typedef shared_ptr< rtc_compute_t > p_rtc_compute_t; 

  struct rtc_func_call_t { 
    string rtc_func_name; 
    // note/FIXME: currently, the split between {in,inout,out}_args is unused.
    vect_string in_args; 
    vect_string inout_args; 
    vect_string out_args; 
    vect_uint32_t u32_args;
    string call_tag;

    zi_uint32_t tpb;
    zi_uint32_t blks;

    uint32_t call_id;
  };
  typedef vector< rtc_func_call_t > vect_rtc_func_call_t; 

}

#endif /* _RTC_COMPUTE_H_ */
