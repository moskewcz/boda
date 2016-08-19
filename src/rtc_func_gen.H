// Copyright (c) 2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RTC_FUNC_GEN_H_
#define _RTC_FUNC_GEN_H_

#include"boda_base.H"
#include"rtc_compute.H"
#include"nesi.H" // for nesi op <<
#include<boost/filesystem.hpp>
#include"lexp.H"
#include"str_util.H"
#include"op_base.H"

namespace boda 
{
  using boost::filesystem::path;

  typedef map< string, uint32_t > map_str_u32_t;
  typedef map< string, float > map_str_float_t;
  typedef map< string, dims_t > map_str_dims_t;

  struct op_base_t;
  struct rtc_call_gen_t;  
  typedef shared_ptr< rtc_call_gen_t > p_rtc_call_gen_t; 
  typedef vector< p_rtc_call_gen_t > vect_p_rtc_call_gen_t; 
  typedef map< op_base_t, p_rtc_call_gen_t > rtc_func_sigs_map_t;

  struct rcg_func_call_t {
    p_rtc_call_gen_t func;
    string call_tag;
    map_str_rtc_arg_t arg_map;
    uint32_t call_id;
  };
  typedef vector< rcg_func_call_t > vect_rcg_func_call_t; 
  typedef shared_ptr< rcg_func_call_t > p_rcg_func_call_t; 
  typedef vector< p_rcg_func_call_t > vect_p_rcg_func_call_t;

  struct custom_codegen_t { 
    virtual void gen_op( rtc_call_gen_t * rcg, string const & op_name ) = 0; 
    virtual ~custom_codegen_t( void ) { }
  };
  typedef shared_ptr< custom_codegen_t > p_custom_codegen_t; 

  // for convenience, declare the factory func for cnn_custom_codegen_t here (code is in cnn_codegen.cc)
  struct cnn_custom_codegen_t; typedef shared_ptr< cnn_custom_codegen_t > p_cnn_custom_codegen_t;
  p_custom_codegen_t make_cnn_custom_codegen_t( void );

  struct arg_decl_t {
    string vn;
    zi_bool multi;
    string tn;
    zi_uint32_t loi;
    string io_type;
    zi_bool dyn;
    vect_dims_t ok_dims;
    void set_vn_tn( string const & vn_, string const & tn_ );
    void ref_parse( string const & line );
    void arg_parse( string const & line );
    uint32_t get_multi_sz( op_base_t const & op ) const;
    string get_multi_vn( uint32_t const mix ) const;
  };
  inline std::ostream & operator << ( std::ostream & out, arg_decl_t const & o ) {
    for( vect_dims_t::const_iterator i = o.ok_dims.begin(); i != o.ok_dims.end(); ++i ) {
      out << strprintf( "%s --- vn=%s tn=%s loi=%s io_type=%s ", str(*i).c_str(), str(o.vn).c_str(), 
                        str(o.tn).c_str(), str(o.loi.v).c_str(), str(o.io_type).c_str() );
    }
    return out;
  }
  struct vect_arg_decl_t : public vector< arg_decl_t > {
    struct multi_iter {
      iterator i;
      iterator e;
      op_base_t const * op;
      uint32_t msz_;
      uint32_t mix;
      string vn_;
      bool at_end( void ) const { return i == e; }
      uint32_t msz( void ) { if( !msz_ ) { msz_ = i->get_multi_sz( *op ); } return msz_; }
      void operator ++( void ) { 
        assert_st( mix < msz() ); 
        if( mix < (msz()-1) ) { ++mix; vn_.clear(); return; }
        msz_ = 0; mix = 0; vn_.clear(); ++i;
      }
      string & vn( void ) { if( vn_.empty() ) { vn_ = i->get_multi_vn( mix ); assert_st( !vn_.empty() ); } return vn_; }
      arg_decl_t & ad( void ) { return *i; } 
    };
    multi_iter multi_begin( op_base_t const * const op ) { return multi_iter{begin(),end(),op,0,0}; }
  };
  
  struct ix_decl_t {
    string ix_vn;
    string arg_vn;
    vect_string use_dims;
  };
  typedef vector< ix_decl_t > vect_ix_decl_t; 

  void insert_nda_dims_sz( map_str_str & mss, string const & nda_vn, dims_t const & dims, bool const & dims_only );
  void insert_nda_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() );
  void insert_nda_dyn_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() );

  dims_t dims_from_nda_spec( string const & tn, string const & nda_spec );


  // this parses and stores a cucl function template for later instantiation. in particular, we process all CUCL
  // declarations and store the results into arg_decls and ix_decls. after init, the contents of this object are
  // constant, and all contents are purly derived from the contents of the template file. this object exists both to
  // encapsulate the CUCL parsing code as well as to avoid needing to re-parse template files.
  struct rtc_template_t {
    string template_fn;
    p_string template_str; // read from file
    vect_arg_decl_t arg_decls; // parsed out of template_str
    vect_ix_decl_t ix_decls; // parsed out of template_str
    vect_string all_tvs; // list of all referenced template variables
    zi_bool has_cucl_arg_info;
    void init( string const & template_fn_ );
    void init_proc_line( string const & line );
    p_op_base_t check_args( op_base_t const & rfs_in );
  };

  typedef shared_ptr< rtc_template_t > p_rtc_template_t; 
  typedef map< string, p_rtc_template_t > rtc_template_map_t;

  // note that while this derives from op_base_t, this represents the 'elaborated' signature, where instead of just
  // containing all the information that defines the semantics of this call, it also contains all the data (redundant,
  // generated) that makes the *instantiation* unique. that is, after init(), combined with the function template, the
  // signature of the rtc_call_gen_t defines a unique block of text for the generated function, using only the
  // str_vals field, strictly for text substitution into the template. the dims_vals field remains unchanged,
  // and isn't used directly in template substition: all the info inside it is flattened into str_vals.

  struct rtc_call_geom_t {
    uint32_t tpb;
    uint32_t blks;    
    static uint32_t const get_default_tpb( void ) { return 256; } // FIXME: make 'reasonable default' tpb configurable/heurisitic
    rtc_call_geom_t( void ) : tpb(0), blks(0) {}
    void maybe_update_for_special_cucl_ixs( string const & ix_vn, dims_t const & ix_dims );
  };

  struct dyn_dim_info_t { 
    string nda_vn; // name inside of arg info struct
    string src_vn; // name of arg that dims are drawn from (without alteration for non-ix dims)
    vect_string use_dims; // use_dims spec for ix (if any, empty to use all dims)
  };
  typedef vector< dyn_dim_info_t > vect_dyn_dim_info_t; 

  struct rtc_call_gen_t {
    op_base_t op;
    p_rtc_template_t rtc_func_template; // post-init, currently used only for dynamic ix_decl processing at call time
    map_str_str tsvs; // template-string-values (set with set() function)
    map_str_str cgs; // similar to tsvs, bulk code-gen sections (set with line() function); merged into tsvs at instantiate
    map_str_dims_t all_ix_dims; // post-processed dims for all CUCL IX's; used (only) in custom codegen
    string rtc_prog_str; // final full program string (result of instantiation)
    zi_bool is_compiled; // flag used by rtc_codegen_t to track which functions have been generated
    // info needed to make call
    string gen_fn; // unique function name at rtc level. although we try to make it readable, name has no used semantics.
    rtc_call_geom_t rtc_call_geom;
    vect_dyn_dim_info_t dyn_vars;
    void line( string const & sn, string const & line );
    void set( string const & var, string const & val );
    dims_t apply_use_dims( dims_t const & ix_arg_dims, vect_string const & use_dims );
    void init( op_base_t const & op_, p_rtc_template_t const & rtc_func_template_, custom_codegen_t * const cc, string const & gen_fn_ );
    dims_t const & get_arg_dims_by_name( string const & arg_vn, string const & err_tag = "TEMPLATE" );
    string cai_tn( void ) const { return "int32_t"; } // type string for cucl arg info vars
    void add_dyn_nda_dims_sz( string const & nda_vn, dims_t const & dims, bool const add_ref_templates );
    void add_arg_info_for_dims( dims_t const & dims, vect_int32_t & cucl_arg_info );
    void instantiate_template( string const & template_str );
    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, rcg_func_call_t & rfc );
  };
  
  struct rtc_codegen_t {
    p_rtc_compute_t rtc;
    void init( p_rtc_compute_t const & rtc_, p_custom_codegen_t const & cc_, 
               rtc_compile_opts_t const & rtc_compile_opts_ ) 
    { 
      rtc = rtc_; 
      cc = cc_;
      rtc_compile_opts = rtc_compile_opts_;
    }

    p_rtc_call_gen_t gen_func( op_base_t const & rfs_full );
    p_rtc_call_gen_t gen_func_override_func_name( string const & func_name, op_base_t & op );

    void run_func( rcg_func_call_t & call );

    void read_rtc_func_sigs( filename_t const & rtc_func_sigs_fn );
    void write_rtc_func_sigs( filename_t const & rtc_func_sigs_fn );

    void clear( void ); // cleanup any not-externally-referenced functions by freeing them at the rtc_compute_t level
    void gc_clear( void ); // (only) if there are 'many' functions currently, call clear()
  protected:
    // note: compile() need not be protected (last i checked), but on the other hand we don't seem to ever need to
    // manually call it. if, for example, one wanted to control having groups of functions in the same modules (or more
    // to the point, *not* in the same module), it might be neccessary to call compile() manually after generating some
    // set of functions.
    void compile( void ); // compile all pending functions together (perhaps into single module at rtc_compute_t level)

    // bound at init() along with rtc, never changed.
    p_custom_codegen_t cc;
    rtc_compile_opts_t rtc_compile_opts;

    // cached templates from files. never cleared (templates assumed to be const).
    rtc_template_map_t rtc_templates;

    // generated-function related data; functions may be released on clear()
    set_string used_names;
    rtc_func_sigs_map_t rtc_func_sigs_map;
    vect_p_rtc_call_gen_t compile_pend;

  };

}
#endif /* _RTC_FUNC_GEN_H_ */
