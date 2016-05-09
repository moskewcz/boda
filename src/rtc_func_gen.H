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
  typedef map< op_base_t, p_rtc_call_gen_t > rtc_func_sigs_map_t;
  typedef map< string, p_rtc_call_gen_t > rtc_func_names_map_t;

  struct rcg_func_call_t {
    string rtc_func_name; 
    string call_tag;
    map_str_str arg_map;
    vect_uint32_t u32_args;
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

  string gen_unused_fn( op_base_t const & op, rtc_func_names_map_t & fns );

  struct arg_decl_t {
    string vn;
    string io_type;
    vect_dims_t ok_dims;
    string multi_base_vn;
  };
  inline std::ostream & operator << ( std::ostream & out, arg_decl_t const & o ) {
    for( vect_dims_t::const_iterator i = o.ok_dims.begin(); i != o.ok_dims.end(); ++i ) {
      out << strprintf( "%s --- vn=%s io_type=%s ", str(*i).c_str(), str(o.vn).c_str(), str(o.io_type).c_str() );
    }
    return out;
  }
  typedef vector< arg_decl_t > vect_arg_decl_t; 
  
  struct ix_decl_t {
    string ix_vn;
    string arg_vn;
    vect_string use_dims;
  };
  typedef vector< ix_decl_t > vect_ix_decl_t; 

  void insert_nda_dims_sz( map_str_str & mss, string const & nda_vn, dims_t const & dims, bool const & dims_only );
  void insert_nda_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() );
  dims_t dims_from_nda_spec( string const & nda_spec );
  
  // this parses and stores a cucl function template for later instantiation. in particular, we process all CUCL
  // declarations and store the results into arg_decls and ix_decls. after init, the contents of this object are
  // constant, and all contents are purly derived from the contents of the template file. this object exists both to
  // encapsulate the CUCL parsing code as well as to avoid needing to re-parse template files.
  struct rtc_template_t {
    string template_fn;
    p_string template_str; // read from file
    vect_arg_decl_t arg_decls; // parsed out of template_str
    vect_ix_decl_t ix_decls; // parsed out of template_str
    void init( string const & template_fn_ ) {
      template_fn = template_fn_;
      string const full_template_fn = (path(py_boda_test_dir()) / "rtc" / (template_fn+".cucl")).string();
      template_str = read_whole_fn( full_template_fn );
      vect_string lines = split( *template_str, '\n' );
      template_str->clear();
      for( vect_string::const_iterator i = lines.begin(); i != lines.end(); ++i ) {
	// find magic CUCL comment (if any) and process it
	template_str->append( *i );
	template_str->append( "\n" );
	string const mmc = get_part_after( *i, "//" );
	vect_string mmc_parts = split_ws( strip_ws( mmc ) );
	if( (mmc_parts.size() > 0) && (mmc_parts[0] == "CUCL" ) ) {
	  if( mmc_parts.size() < 2 ) { rt_err( "invalid CUCL magic comment. missing directive after CUCL. saw: " + *i ); }
	  string cd = mmc_parts[1];
	  bool const multi = maybe_strip_suffix( cd, "_MULTI" );
	  if( (cd == "IN") || (cd == "INOUT") || (cd == "OUT") || (cd == "REF") || (cd == "IX") ) { 
	    if( cd == "IX" ) {
	      if( multi ) { rt_err( "invalid use of _MULTI suffix on CUCL IX decl" ); }
	      if( mmc_parts.size() < 4 ) { rt_err( "invalid CUCL IX decl; missing ix_name and/or arg_name: " + *i ); }
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
	      if( mmc_parts.size() < 3 ) { rt_err( "invalid CUCL IN/INOUT/OUT annotation; missing dims spec: " + *i ); }
	      // get var name 
	      vect_string const arg_decl = split_ws( strip_ws( replace_chars_with_char( get_part_before( *i, "//" ), ",);{*/", ' ' ) ) );
	      vect_dims_t ok_dims;
	      for( uint32_t i = 2; i != mmc_parts.size(); ++i ) { ok_dims.push_back( dims_from_nda_spec( mmc_parts[i] ) );}
	      string const arg_name = arg_decl.back();
	      string multi_base_vn; // default -> empty/not-multi
	      if( multi ) {
		multi_base_vn = arg_name;
		bool const did_strip = maybe_strip_suffix( multi_base_vn, "_N" );
		if( !did_strip ) { rt_err("MULTI arg names must end with _N (to be _0, _1, ..."); }
	      }
	      arg_decls.push_back( arg_decl_t{ arg_name, cd, ok_dims, multi_base_vn } );
	    }
	  } else if( cd == "INCLUDE" ) {
	    // note -- currently, include files are inserted verbatim, and cannot contain CUCL magic-comments (including
	    // includes) themselves
	    if( mmc_parts.size() != 3 ) { rt_err( "invalid CUCL INCLUDE decl; must be exactly CUCL INCLUDE filename.h; saw: " + *i);}
	    string const full_include_fn = (path(py_boda_test_dir()) / "rtc" / mmc_parts[2]).string();
	    template_str->append( *read_whole_fn( full_include_fn ) );
	  } else { rt_err( "invalid CUCL directive '"+cd+"'. saw:" + *i ); }
	}
      }
    }

    // this takes an input func signature and checks it against this template. it produces two things: (1) a set of
    // arg_decls with any multis flattened out (based on the args in the input signature). (2) a reduced function
    // signature with only the arg dims needed/used by this template. this reduced signature is suitable for uniqueing
    // the semantics of this template wrt the input signature -- that is, input signatures that differ only in dims that
    // are not used by the template won't have differing semantics.
    p_op_base_t check_args( op_base_t const & rfs_in, vect_arg_decl_t & flat_arg_decls ) {
      p_op_base_t rfs_out( new op_base_t( rfs_in ) );
      rfs_out->dims_vals.clear(); // for now: keep all of rfs_in expect dims_vals (which we reduce to only the used elements)
      assert_st( flat_arg_decls.empty() );
      string arg_check_error;

      // process/flatten-out multi args
      for( vect_arg_decl_t::const_iterator i = arg_decls.begin(); i != arg_decls.end(); ++i ) {
	if( i->multi_base_vn.empty() ) { flat_arg_decls.push_back( *i ); }
	else {
	  uint32_t const multi_sz = rfs_in.get_u32( i->multi_base_vn + "_num" );
	  for( uint32_t mix = 0; mix != multi_sz; ++mix ) {
	    string const multi_arg_name = i->multi_base_vn + "_" + str(mix);
	    // note: multi-derived args are (still) marked multi=1 in flat_arg_decls. this is used to later
	    // code-generate the decl section for these args.
	    flat_arg_decls.push_back( *i ); 
	    flat_arg_decls.back().vn = multi_arg_name;
	  }
	}
      }
      
      for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
	uint32_t const ix = i - flat_arg_decls.begin();
	map_str_dims_t::const_iterator adi = rfs_in.dims_vals.find( i->vn );
	if( adi == rfs_in.dims_vals.end() ) {
	  arg_check_error +=  strprintf( "referenced %s arg '%s' not present in dims_vals; ", 
					 i->io_type.c_str(), i->vn.c_str() );
	  continue;
	}
	dims_t const & arg_dims = adi->second;
	if( !arg_dims.has_name() ) { arg_check_error += "call arg "+str(ix)+ " must have names for all dims; "; }
	bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
	if( !dims_only && arg_dims.has_padding() ) { 
	  arg_check_error += "call args "+str(ix)+ " must not have padding; "; } // FIXME: maybe too strong

	bool matches_decl = 0;
	for( uint32_t j = 0; j != i->ok_dims.size(); ++j ) {
	  if( arg_dims == dims_t() ) { matches_decl = 1; } // NULL case
	  if( arg_dims.matches_template( i->ok_dims[j] ) ) { matches_decl = 1; }
	}
	if( !matches_decl ) { arg_check_error += "call arg "+str(ix)+" incompatible with decl arg "
	    "(dim count mismatch or dim (non-zero and thus req'd) name/size/stride mismatch; "; }
	must_insert( rfs_out->dims_vals, i->vn, arg_dims ); // keep exactly used dims_vals in signature
      }
      if( !arg_check_error.empty() ) {
	string arg_err = "RTC template function instantiation argument error: " + template_fn + ": " + arg_check_error + "\n";
	for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
	  arg_err += strprintf( "ARG[%s]:\n", str(i - flat_arg_decls.begin()).c_str() );
	  arg_err += strprintf( "  DECL: %s\n", str(*i).c_str() );
	  map_str_dims_t::const_iterator adi = rfs_in.dims_vals.find( i->vn );
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
  struct rtc_call_gen_t : public op_base_t {
    vect_arg_decl_t flat_arg_decls; // copy of rtc_func_template->arg_decls, but with multis flattened out
    map_str_dims_t all_ix_dims;
    map_str_str cgs; // code-gen sections
    string rtc_prog_str; // final full program string
    
    // info needed to make call
    string gen_fn;
    bool has_final_flags_arg;
    uint32_t tpb;
    uint32_t blks;

    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, rcg_func_call_t & rfc, uint32_t const & flags );

    void line( string const & sn, string const & line ) { 
      string & cg = cgs[sn];
      if(cg.empty()) { cg += "// begin "+sn+"\n"; } // adding first line in (new) section, add header
      cg += "   " + line + "\n"; 
    }
    void set( string const &var, string const &val ) { must_insert( str_vals, var, val ); }

    dims_t const & get_arg_dims_by_name( string const & arg_vn, string const & err_tag = "TEMPLATE" ) {
      map_str_dims_t::const_iterator i = dims_vals.find( arg_vn );
      if( i == dims_vals.end() ) { 
	rt_err( strprintf( "referenced %s arg '%s' not present in dims_vals", err_tag.c_str(), arg_vn.c_str() ) );
      }
      return i->second;
    }

    rtc_call_gen_t( op_base_t const & rfs_ ) : op_base_t( rfs_ ) { }

    void init( p_rtc_template_t const & rtc_func_template, vect_arg_decl_t const & flat_arg_decls_, 
	       custom_codegen_t * const custom_codegen, string const & gen_fn_ ){
      gen_fn = gen_fn_;
      has_final_flags_arg = 0;
      set( "rtc_func_name", gen_fn );
      flat_arg_decls = flat_arg_decls_;

      tpb = 0;
      blks = 0;
      uint32_t const default_tpb = 256; // FIXME: make 'reasonable default' tpb configurable/heurisitic
      for( vect_ix_decl_t::const_iterator i = rtc_func_template->ix_decls.begin(); i != rtc_func_template->ix_decls.end(); ++i ) {
	dims_t const & ix_arg_dims = get_arg_dims_by_name( i->arg_vn, "IX" );
	if( !ix_arg_dims.has_sz_and_stride_and_name() ) { rt_err( "NEVER_SAY_NEVER, but can't create CUCL IX for dynamically-sized var" ); }
	dims_t ix_dims;
	if( i->use_dims.empty() ) { ix_dims = ix_arg_dims; } 
	else {
	  for( vect_string::const_iterator j = i->use_dims.begin(); j != i->use_dims.end(); ++j ) {
	    dim_t const * use_dim = ix_arg_dims.get_dim_by_name( *j );
	    if( !use_dim ) { rt_err( "specified use_dim '"+*j+"' not found in target arg's dims" ); }
	    ix_dims.push_back( *use_dim );
	  }
	}
	ix_dims.calc_strides(); // note: stride are garbage prior to this call (which is okay)
	must_insert( all_ix_dims, i->ix_vn, ix_dims );
	insert_nda_ix_exprs( str_vals, i->ix_vn, ix_dims );
	// special cases for index var names
	if( i->ix_vn == "GLOB_ID_1D" ) { 
	  // if GLOB_ID_1D is an index for some arg, assume we want 1 thread per element of that arg, and assume block
	  // size doesn't matter, so use a reasonable default.
	  if( tpb || blks ) { rt_err( "CUCL error: GLOB_ID_1D IX encoutered after setting tpb or blks (some other way(s))" ); }
	  tpb = default_tpb;
	  blks = u32_ceil_div( ix_dims.dims_prod(), tpb );
	} else if( i->ix_vn == "GRP_ID_1D" ) {
	  if( blks ) { rt_err( "CUCL error: GRP_ID_1D IX encoutered after setting blks (some other way)" ); }
	  blks = ix_dims.dims_prod();
	} else if( i->ix_vn == "LOC_ID_1D" ) {
	  if( tpb ) { rt_err( "CUCL error: LOC_ID_1D IX encoutered after setting tpb (some other way)" ); }
	  tpb = ix_dims.dims_prod();
	}
      }
      if( !tpb ) { tpb = default_tpb; } // if not set, use a reasonable default
      // assert_st( rf->blks ); // too strong. if not set, dynamic # blks case
      
      if( custom_codegen ) { custom_codegen->gen_op( this, type ); }

      // make these always availible as template vars, since why not?
      set( "tpb", str(tpb) ); // should always be fixed/constant/valid (i.e. gen'd kernels never have dynamic tpb)
      set( "blks", str(blks) ); // may be 0 if # of blocks is dynamic
      set( "warp_sz", str("UNKNOWN") ); // yeah, not the best, but probably not exactly wrong. don't use it for real

      instantiate_template( *rtc_func_template->template_str );
    }



    void instantiate_template( string const & template_str ) {
      assert_st( rtc_prog_str.empty() ); // should only call only
      // generate arg decl code for multi args
      for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
	if( !i->multi_base_vn.empty() ) { line( i->multi_base_vn + "_decl", "GASQ float const * const " + i->vn + "," ); }
	dims_t const & arg_dims = get_arg_dims_by_name( i->vn );
	bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
	insert_nda_dims_sz( str_vals, i->vn, arg_dims, dims_only );
      }

      rtc_prog_str += "// -- codegen begins for '"+type+
	"'; template substituion table used (bulk sections ommited): --\n";
      for( map_str_str::const_iterator i = str_vals.begin(); i != str_vals.end(); ++i ) {
	rtc_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
      }
      for( map_str_str::iterator i = cgs.begin(); i != cgs.end(); ++i ) { // terminate and emit bulk cgs
	i->second += "    // end "+i->first;
	set( i->first, i->second ); 
      } 
      lexp_name_val_map_t tf_nvm{ p_lexp_t() };
      tf_nvm.insert_leafs_from( str_vals );
      string rtc_func_str;
      try {
	str_format_from_nvm( rtc_func_str, template_str, tf_nvm );
      } catch( rt_exception const & rte ) {
	printf( "rfs=%s\n", str((*this)).c_str() );
	rt_err( strprintf( "instantiation failed; see above; type=%s; error was: %s\n", type.c_str(), rte.err_msg.c_str() ) );
      }
      rtc_prog_str += rtc_func_str;      
    }

  };
  
  struct rtc_codegen_t {
    p_rtc_compute_t rtc;
    void init( p_rtc_compute_t const & rtc_ ) { rtc = rtc_; }
    rtc_template_map_t rtc_templates;

    rtc_func_names_map_t rtc_func_names_map;
    rtc_func_sigs_map_t rtc_func_sigs_map;

    vect_rtc_func_info_t rtc_prog_str_infos;
    string rtc_prog_str;

    string gen_func( custom_codegen_t * const cc, op_base_t const & rfs_full );
    void run_rfc( bool const & show_rtc_calls, rcg_func_call_t & rfc, uint32_t const & flags ) {    
      must_find( rtc_func_names_map, rfc.rtc_func_name )->run_rfc( rtc, show_rtc_calls, rfc, flags );
    }

    void compile( bool const show_compile_log, bool const enable_lineinfo, bool const show_func_attrs );

    void read_rtc_func_sigs( filename_t const & rtc_func_sigs_fn );
    void write_rtc_func_sigs( filename_t const & rtc_func_sigs_fn );

    void clear( void );

  };

}
#endif /* _RTC_FUNC_GEN_H_ */
