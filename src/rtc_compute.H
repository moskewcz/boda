// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RTC_COMPUTE_H_
#define _RTC_COMPUTE_H_

#include"boda_base.H"
#include"op_base.H"

namespace boda 
{
  struct rtc_func_info_t {
    string func_name;
    op_base_t op;
  };
  typedef vector< rtc_func_info_t > vect_rtc_func_info_t; 

  struct rtc_func_call_t;

  struct rtc_compute_t : virtual public nesi // NESI(help="modes that can be used for rtc",is_abstract=1,tid_vn="be")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string be; //NESI(help="name of back-end to use for rtc",req=1)
    uint32_t gen_src; //NESI(default=0,help="if 1, output generated src files just before compilation")
    filename_t gen_src_output_dir; //NESI(default="%(boda_output_dir)/rtc-gen-src", help="if requested, directory where generated source files will be placed")

    virtual void init( void ) = 0;

    virtual void create_var_with_dims( string const & vn, dims_t const & dims ) = 0;
    virtual void release_var( string const & vn ) = 0;
    virtual dims_t get_var_dims( string const & vn ) = 0;
    virtual void set_var_to_zero( string const & vn ) = 0;
    
    // note: must specify all functions that will ever be called from this source block at compile time.
    virtual void compile( string const & src, bool const show_compile_log, bool const enable_lineinfo,
			  vect_rtc_func_info_t const & func_infos, bool const show_func_attrs ) = 0;
    // note: run() overwrites / fills in the call_id field. it is valid after run() returns until realease_per_call_id_data() is called.
    virtual void run( rtc_func_call_t & rfc ) = 0; 
    virtual void finish_and_sync( void ) = 0;
    virtual void release_per_call_id_data( void ) = 0; // invalidates all call_ids inside rtc_func_call_t's
    virtual void release_all_funcs( void ) = 0; 

    // get duration from start of call b to end of call e (where b may == e to time a single call).

    // we almost-only need to get the timing of a single call, but we allow spanning calls for convenience and/or to allow
    // detecting inconsistencies/overheads between the sum of per-call times and the overall all-calls time (as proxied by
    // calling get_dur( first_call, last_call ) ).
    // note: return value is in milliseconds (ms)
    virtual float get_dur( uint32_t const & b, uint32_t const & e ) = 0; // note: uses call_id field, which must be valid

    virtual void profile_start( void ) = 0;
    virtual void profile_stop( void ) = 0;

    // non-virtual factored-out conversion functions layered on top of copy_{to,from}_var() and related functions
    // vect_float <-> var conversions/copies
    void init_var_from_vect_float( string const & vn, vect_float const & v );
    void set_vect_float_from_var( vect_float & v, string const & vn );
    virtual void copy_var_to_nda( p_nda_t const & nda, string const & vn ) = 0;
    virtual void copy_nda_to_var( string const & vn, p_nda_t const & nda ) = 0;

    p_nda_t copy_var_as_flat_nda( string const & vn );
    void copy_ndas_to_vars( vect_string const & names, map_str_p_nda_t const & ndas );
    void copy_vars_to_ndas( vect_string const & names, map_str_p_nda_t & ndas );

    // FIXME_TNDA: for now, allow these wrappers to use nda_float_t (due to the 'can't pass container-of-derived for
    // container-of-base' language issue)
    void copy_ndas_to_vars( vect_string const & names, map_str_p_nda_float_t const & ndas );
    void copy_vars_to_ndas( vect_string const & names, map_str_p_nda_float_t & ndas );

    void create_var_from_nda( p_nda_t const & nda, string const & vn );
    p_nda_t create_nda_from_var( string const & vn );

  };
  struct rtc_compute_t; typedef shared_ptr< rtc_compute_t > p_rtc_compute_t; 

  struct rtc_func_call_t { 
    string rtc_func_name; 
    // note/FIXME: currently, the split between {in,inout,out}_args is unused.
    vect_string in_args; 
    vect_string inout_args; 
    vect_string out_args; 
    vect_uint32_t u32_args;
    zi_bool has_cucl_arg_info;
    vect_int32_t cucl_arg_info;
    string call_tag;

    zi_uint32_t tpb;
    zi_uint32_t blks;

    uint32_t call_id;
  };
  typedef vector< rtc_func_call_t > vect_rtc_func_call_t; 

  // FIXME: things shared by backends, but not used (should not be used?) by clients. move to shared-by-backends header?
  void rtc_launch_check_blks_and_tpb( std::string const & rtc_func_name, uint64_t const blks, uint64_t const tpb );

}

#endif /* _RTC_COMPUTE_H_ */
