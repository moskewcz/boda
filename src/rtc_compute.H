// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RTC_COMPUTE_H_
#define _RTC_COMPUTE_H_

#include"boda_base.H"
#include"op_base.H"

namespace boda 
{
  struct rtc_compile_opts_t : virtual public nesi // NESI(help="runtime compilation options") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t show_compile_log; //NESI(default=0,help="if 1, print compilation log")
    uint32_t enable_lineinfo; //NESI(default=0,help="if 1, enable lineinfo for runtime compilation")
    uint32_t show_func_attrs; //NESI(default=0,help="if 1, print func attrs after load")

    // not used at compile time, and could vary per-call. but we seem to always have it at compile time, and we never
    // change it, so it's okay to put here for now.
    uint32_t show_rtc_calls; //NESI(default=0,help="if 1, print every rtc call (at call time)")

    rtc_compile_opts_t( void ) : show_compile_log(0), enable_lineinfo(0), show_func_attrs(0), show_rtc_calls(0) { }
  };

  struct rtc_func_info_t {
    string func_name;
    string func_src;
    vect_string arg_names;
    op_base_t op;
  };
  typedef vector< rtc_func_info_t > vect_rtc_func_info_t; 
  std::ostream & operator << ( std::ostream & out, rtc_func_info_t const & o );

  struct rtc_func_call_t;

  struct rtc_compute_t : virtual public nesi // NESI(help="modes that can be used for rtc",is_abstract=1,tid_vn="be")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string be; //NESI(help="name of back-end to use for rtc",req=1)
    uint32_t gen_src; //NESI(default=0,help="if 1, output generated src files just before compilation")
    filename_t gen_src_output_dir; //NESI(default="%(boda_output_dir)/rtc-gen-src", help="if requested, directory where generated source files will be placed")
    uint32_t use_platform_id; //NESI(default=0,help="ID of the platform to use")
    uint32_t use_device_id; //NESI(default=0,help="ID of the device to use")

    string rtc_base_decls;
    string const & get_rtc_base_decls( void ); // note: caches results in rtc_base_decls

    virtual void init( void ) = 0;
    virtual string get_plat_tag( void ) = 0;

    virtual void create_var_with_dims( string const & vn, dims_t const & dims ) = 0;
    virtual void create_var_with_dims_as_reshaped_view_of_var( string const & vn, dims_t const & dims, string const & src_vn ) = 0; // note: will point to (and share as a peer) the data of existing var named by src_vn
    virtual void release_var( string const & vn ) = 0;
    virtual dims_t get_var_dims( string const & vn ) = 0;
    virtual void set_var_to_zero( string const & vn ) = 0;
    
    // note: must specify all functions that will ever be called from this source block at compile time.
    virtual void compile( vect_rtc_func_info_t const & func_infos, rtc_compile_opts_t const & opts ) = 0;
    virtual void release_func( std::string const & func_name ) = 0;

    // note: run() overwrites / fills in the call_id field. it is valid after run() returns until realease_per_call_id_data() is called.
    virtual uint32_t run( rtc_func_call_t const & rfc ) = 0; 
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
    virtual p_nda_t get_var_raw_native_pointer( string const & vn ) = 0; // for integration with external low-level libs
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
  typedef vector< p_rtc_compute_t > vect_p_rtc_compute_t;
  // an rtc argument is either: (1) the name of a var (stored in n, v is null) or (2) a value (stored in v, n is
  // empty). for var arguments, a pointer to the on-device memory for that var is passed. for values, the entire value
  // is passed as a single by-value argument (i.e. as raw data).
  struct rtc_arg_t {
    rtc_arg_t( void ) {}
    rtc_arg_t( string const & n_ ) : n(n_) {}
    rtc_arg_t( p_nda_t const & v_ ) : v(v_) {}
    string n;
    p_nda_t v;
    bool is_valid( void ) const { return bool(v) != (!n.empty()); }
    bool is_var( void ) const { assert_st( is_valid() ); return !n.empty(); }
    bool is_nda( void )const { assert_st( is_valid() ); return bool(v); }
    string const & get_var( void ) const { assert_st( is_var() ); return n; }
    p_nda_t const & get_nda( void ) const { assert_st( is_nda() ); return v; }
    dims_t get_dims( rtc_compute_t & rtc ) const { return is_nda() ? v->dims : rtc.get_var_dims( n ); }
  };
  std::ostream & operator << ( std::ostream & out, rtc_arg_t const & o );
  typedef vector< rtc_arg_t > vect_rtc_arg_t; 
  typedef map< string, rtc_arg_t > map_str_rtc_arg_t; 

  struct rtc_func_call_t { 
    string rtc_func_name; 
    map_str_rtc_arg_t arg_map;

    zi_uint32_t tpb;
    zi_uint32_t blks;
  };
  typedef vector< rtc_func_call_t > vect_rtc_func_call_t; 

  // FIXME: things shared by backends, but not used (should not be used?) by clients. move to shared-by-backends header?
  void rtc_launch_check_blks_and_tpb( std::string const & rtc_func_name, uint64_t const blks, uint64_t const tpb );
  void rtc_reshape_check( dims_t const & dims, dims_t const & src_dims );
}

#endif /* _RTC_COMPUTE_H_ */
