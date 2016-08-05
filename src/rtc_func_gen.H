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
    map_str_str arg_map;
    vect_p_nda_t nda_args;
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
    void set_vn_tn( string const & vn_, string const & tn_ ) {
      vn = vn_;
      tn = get_part_before( tn_, "_multi" );
      multi.v = (tn != tn_); // if we found _multi, we stripped it off, so tn is shorter than tn_
    }
    void ref_parse( string const & line ) {
      vect_string const arg_decl = split_ws( strip_ws( strip_ending_chars( get_part_before( line, "//" ), " */" ) ) );
      if( arg_decl.size() < 1 ) { rt_err( "invalid CUCL REF decl; no var name present:" + line ); }
      set_vn_tn( arg_decl.back(), string() ); // no tn (type name) for ref decls
    }
    void arg_parse( string const & line ) {
      vect_string const arg_decl = split_ws( strip_ws( strip_ending_chars( get_part_before( line, "//" ), " ,);{" ) ) );
      if( arg_decl.size() < 1 ) { rt_err( "invalid CUCL io var decl; no var name found:" + line ); }
      for( vect_string::const_reverse_iterator i = arg_decl.rbegin()+1; i != arg_decl.rend(); ++i ) {
        if( (*i) == "*" ) { ++loi.v; continue; }
        if( (*i) == "const" ) { continue; } // FIXME: for now, ignore const?
        set_vn_tn( arg_decl.back(), *i ); break;
      }
    }
    uint32_t get_multi_sz( op_base_t const & op ) const { return multi.v ? op.get_u32( vn+"_num" ) : 1; }
    string get_multi_vn( uint32_t const mix ) const { 
      if( multi.v ) { return vn + "_" + str(mix); }
      assert_st( mix == 0 ); return vn;
    }
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
    void init( string const & template_fn_ ) {
      template_fn = template_fn_;
      string const full_template_fn = (path(py_boda_test_dir()) / "rtc" / (template_fn+".cucl")).string();
      template_str = read_whole_fn( full_template_fn );
      vect_string lines = split( *template_str, '\n' );
      template_str->clear();
      for( vect_string::const_iterator i = lines.begin(); i != lines.end(); ++i ) {
        try { init_proc_line( *i ); }
        catch( rt_exception & rte ) {
          rte.err_msg = strprintf( "Error parsing CUCL template from file %s on line %s:\n--> %s\n%s", 
                                   str(full_template_fn).c_str(), str((i-lines.begin())+1).c_str(), 
                                   (*i).c_str(),
                                   rte.err_msg.c_str() );
          throw;
        }
      }
    }
    
    void init_proc_line( string const & line ) {
      // find any template variable references and record them
      str_format_find_all_refs( all_tvs, line );
      // find magic CUCL comment (if any) and process it
      template_str->append( line );
      template_str->append( "\n" );
      string const mmc = get_part_after( line, "//" );
      vect_string mmc_parts = split_ws( strip_ws( mmc ) );
      if( (mmc_parts.size() > 0) && (mmc_parts[0] == "CUCL" ) ) {
        if( mmc_parts.size() < 2 ) { rt_err( "invalid CUCL magic comment. missing directive after CUCL." ); }
        string cd = mmc_parts[1];
        bool const dyn = maybe_strip_suffix( cd, "_DYN" );
        if( (cd == "IN") || (cd == "INOUT") || (cd == "OUT") || (cd == "REF") || (cd == "IX") ) { 
          if( cd == "IX" ) {
            if( dyn ) { rt_err( "invalid use of _DYN suffix on CUCL IX decl" ); }
            if( mmc_parts.size() < 4 ) { rt_err( "invalid CUCL IX decl; missing ix_name and/or arg_name." ); }
            string const ix_name = mmc_parts[2];
            string const arg_name = mmc_parts[3];
            ix_decls.push_back( ix_decl_t{ ix_name, arg_name } );
            for( uint32_t i = 4; i != mmc_parts.size(); ++i ) {	
              vect_string const opt_parts = split( mmc_parts[i], '=' );
              if( opt_parts.size() != 2 ) { rt_err( "invalid CUCL IX decl option '"+mmc_parts[i]+"', should have exactly 2 '=' seperated parts" ); }
              else if( opt_parts[0] == "use_dims" ) { ix_decls.back().use_dims = split( opt_parts[1], ':' ); }
              else { rt_err( "invalid CUCL IX decl option '"+opt_parts[0]+"'. known opts: use_dims" ); }
            }
          } else {
            if( mmc_parts.size() < 3 ) { rt_err( "invalid CUCL IN/INOUT/OUT annotation; missing dims spec." ); }
            arg_decl_t cad;
            cad.io_type = cd;
            cad.dyn.v = dyn;
            if( cd == "REF" ) {
              cad.ref_parse( line );
            } else {
              cad.arg_parse( line );
              if( cad.tn.empty() ) { rt_err( "invalid CUCL io var decl; no var type found." ); }
              if( cad.loi.v != 1 ) { rt_err( "invalid CUCL io var decl; should be exactly one level-of-indirection/*.");}
              // special case: if type is CUCL type of this var, then there is no restriction on the type that can be
              // passed for this var. note that the only supported types are basic types and this case; there's no
              // ability to use the type of other vars or some other string template or the like to restrict/set the
              // type.
              if( cad.tn == ("%("+cad.vn+"_tn)") ) { cad.tn = ""; } 
            }
            for( uint32_t i = 2; i != mmc_parts.size(); ++i ) { 
              cad.ok_dims.push_back( dims_from_nda_spec( cad.tn, mmc_parts[i] ) );
            }
            arg_decls.push_back( cad );
          }
        } else if( cd == "ARGINFO" ) {
          if( has_cucl_arg_info.v ) { rt_err( "duplicate CUCL ARGINFO declartion. must have exactly 0 or 1 ARGINFO arguments."); }
          has_cucl_arg_info.v = 1;
          arg_decl_t cad;
          cad.arg_parse( line );
          if( cad.multi.v ) { rt_err( "invalid CUCL ARGINFO decl; should not be multi." ); }
          if( cad.tn.empty() ) { rt_err( "invalid CUCL ARGINFO decl; no var type found." ); }
          if( cad.tn != "%(rtc_func_name)_arg_info_t" ) { rt_err( "invalid CUCL ARGINFO decl; must use '%(rtc_func_name)_arg_info_t' as type name." ); }
          if( cad.vn != "cucl_arg_info" ) { rt_err( "invalid CUCL ARGINFO decl; must use 'cucl_arg_info' as var name." ); }
          if( cad.loi.v != 0 ) { rt_err( "invalid CUCL ARGINGO decl; should be exactly zero levels-of-indirection/*s:");}  
          
        } else if( cd == "INCLUDE" ) {
          // note -- currently, include files are inserted verbatim, and cannot contain CUCL magic-comments (including
          // includes) themselves
          if( mmc_parts.size() != 3 ) { rt_err( "invalid CUCL INCLUDE decl; must be exactly CUCL INCLUDE filename.h.");}
          string const full_include_fn = (path(py_boda_test_dir()) / "rtc" / mmc_parts[2]).string();
          template_str->append( *read_whole_fn( full_include_fn ) );
        } else { rt_err( "invalid CUCL directive '"+cd+"'." ); }
      }
    }

    // this takes an input func signature and checks it against this template. it produces a reduced function
    // signature with only the arg dims needed/used by this template. this reduced signature is suitable for uniqueing
    // the semantics of this template wrt the input signature -- that is, input signatures that differ only in dims that
    // are not used by the template won't have differing semantics.
    p_op_base_t check_args( op_base_t const & rfs_in ) {
      p_op_base_t rfs_out( new op_base_t( rfs_in ) );
      rfs_out->dims_vals.clear(); // for now: keep all of rfs_in expect dims_vals (which we reduce to only the used elements)
      string arg_check_error;

      // error check existence of num field for multi args
      for( vect_arg_decl_t::multi_iter i = arg_decls.multi_begin( &rfs_in ); !i.at_end(); ++i ) {
	if( i.ad().multi.v ) {
          string const multi_num = i.ad().vn+"_num";
	  if( !has(rfs_in.str_vals,multi_num) ) {
            arg_check_error += strprintf( "mult arg '%s' missing num var '%s' in str_vals; ", 
                                          i.ad().vn.c_str(), multi_num.c_str() );
            i.msz_ = 1; // note: yikes! a bit hacky, we need to skip the entire multi, so fake-set msz=1 and continue.
            continue;
          }
        }
        map_str_dims_t::const_iterator adi = rfs_in.dims_vals.find( i.vn() );
        if( adi == rfs_in.dims_vals.end() ) {
          arg_check_error +=  strprintf( "referenced %s arg '%s' not present in dims_vals; ", 
                                         i.ad().io_type.c_str(), i.vn().c_str() );
          continue;
        }
        dims_t const & arg_dims = adi->second;
        if( !arg_dims.has_name() ) { arg_check_error += "call arg '"+i.vn()+"' must have names for all dims; "; }
        bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
        if( !dims_only && arg_dims.has_padding() ) { 
          arg_check_error += "call arg '"+i.vn()+"' must not have padding; "; } // FIXME: maybe too strong
        bool matches_decl = 0;
        for( uint32_t j = 0; j != i.ad().ok_dims.size(); ++j ) {
          if( arg_dims == dims_t() ) { matches_decl = 1; break; } // NULL case
          if( arg_dims.matches_template( i.ad().ok_dims[j] ) ) { matches_decl = 1; break; }
        }
        if( !matches_decl ) { arg_check_error += "call arg '"+str(i.vn())+"' incompatible with decl arg "
            "(dim count mismatch or dim (non-zero and thus req'd) name/size/stride mismatch; "; }
        if( i.ad().dyn.v ) {  // if dynamic, zero out sizes/stride, since gen'd func will work for all sizes/strides
          dims_t arg_dims_no_sizes_or_strides = arg_dims;
          arg_dims_no_sizes_or_strides.clear_strides();
          arg_dims_no_sizes_or_strides.clear_dims();
          must_insert( rfs_out->dims_vals, i.vn(), arg_dims_no_sizes_or_strides ); 
        } else {
          must_insert( rfs_out->dims_vals, i.vn(), arg_dims ); // keep exactly used dims_vals in signature
        }
      }
      if( !arg_check_error.empty() ) {
	string arg_err = "RTC template function instantiation argument error: " + template_fn + ": " + arg_check_error + "\n";
	for( vect_arg_decl_t::multi_iter i = arg_decls.multi_begin( &rfs_in ); !i.at_end(); ++i ) {
          arg_err += strprintf( "ARG[%s]:\n", i.vn().c_str() );
          arg_err += strprintf( "  DECL: %s\n", str(i.ad()).c_str() );
          map_str_dims_t::const_iterator adi = rfs_in.dims_vals.find( i.vn() );
          if( adi != rfs_in.dims_vals.end() ) { arg_err += strprintf( "  CALL: %s\n", str(adi->second).c_str() ); }
          else { arg_err += "  CALL: not found in ref dims.\n"; }
	}
	arg_err += "full rfs: " + str(rfs_in) + "\n";
	rt_err( arg_err );
      }
      return rfs_out;
    }
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
    static uint32_t const default_tpb = 256; // FIXME: make 'reasonable default' tpb configurable/heurisitic
    rtc_call_geom_t( void ) : tpb(0), blks(0) {}
    void maybe_update_for_special_cucl_ixs( string const & ix_vn, dims_t const & ix_dims ) {
      // special cases for index var names
      if( ix_vn == "GLOB_ID_1D" ) { 
        // if GLOB_ID_1D is an index for some arg, assume we want 1 thread per element of that arg, and assume block
        // size doesn't matter, so use a reasonable default if it's not already set.
        if( !tpb ) { tpb = default_tpb; }
        if( blks ) { rt_err( "CUCL error: GLOB_ID_1D IX encoutered after setting blks (some other way)");}
        blks = u32_ceil_div( ix_dims.dims_prod(), tpb );
      } else if( ix_vn == "GRP_ID_1D" ) {
        if( blks ) { rt_err( "CUCL error: GRP_ID_1D IX encoutered after setting blks (some other way)" ); }
        blks = ix_dims.dims_prod();
      } else if( ix_vn == "LOC_ID_1D" ) {
        if( tpb ) { rt_err( "CUCL error: LOC_ID_1D IX encoutered after setting tpb (some other way)" ); }
        tpb = ix_dims.dims_prod();
      }
    }
  };

  struct dyn_dim_info_t { 
    string nda_vn; // name inside of arg info struct
    string src_vn; // name of arg that dims are drawn from (without alteration for non-ix dims)
    vect_string use_dims; // use_dims spec for ix (if any, empty to use all dims)
  };
  typedef vector< dyn_dim_info_t > vect_dyn_dim_info_t; 

  struct rtc_call_gen_t : public op_base_t {
    map_str_dims_t all_ix_dims;
    map_str_str cgs; // code-gen sections
    string rtc_prog_str; // final full program string
    zi_bool is_compiled; // flag used by rtc_codegen_t to track which functions have been generated
    // info needed to make call
    string gen_fn;
    bool has_final_flags_arg;
    rtc_call_geom_t rtc_call_geom;
    p_rtc_template_t rtc_func_template; // post-init, currently used only for dynamic ix_decl processing at call time
    vect_dyn_dim_info_t dyn_vars;
    void line( string const & sn, string const & line );
    void set( string const &var, string const &val );
    dims_t apply_use_dims( dims_t const & ix_arg_dims, vect_string const & use_dims );
    rtc_call_gen_t( op_base_t const & rfs_ ) : op_base_t( rfs_ ) { } // FIXME: move op_base_t setting to init()?
    void init( p_rtc_template_t const & rtc_func_template_, custom_codegen_t * const cc, string const & gen_fn_ );
    dims_t const & get_arg_dims_by_name( string const & arg_vn, string const & err_tag = "TEMPLATE" );
    string cai_tn( void ) const { return "int32_t"; } // type string for cucl arg info vars
    void add_dyn_nda_dims_sz( string const & nda_vn, dims_t const & dims, bool const add_ref_templates );
    void add_arg_info_for_dims( dims_t const & dims, vect_int32_t & cucl_arg_info );
    void instantiate_template( string const & template_str );
    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, rcg_func_call_t & rfc, uint32_t const & flags);
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
    void run_func( rcg_func_call_t & call, uint32_t const & flags );

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
