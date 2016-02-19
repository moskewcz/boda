// Copyright (c) 2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RTC_FUNC_GEN_H_
#define _RTC_FUNC_GEN_H_

#include"boda_base.H"
#include"rtc_compute.H"

namespace boda 
{
  using boost::filesystem::path;

  typedef map< string, uint32_t > map_str_u32_t;
  typedef map< string, float > map_str_float_t;
  typedef map< string, dims_t > map_str_dims_t;

  struct rtc_func_sig_t;
  struct rtc_call_gen_t;  
  typedef shared_ptr< rtc_call_gen_t > p_rtc_call_gen_t; 
  typedef map< rtc_func_sig_t, p_rtc_call_gen_t > rtc_func_sigs_map_t;
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

  struct custom_codegen_t { virtual void gen_op( rtc_call_gen_t * rcg, string const & op_name ) = 0; };

  struct rtc_func_sig_t { 
    string fn;
    map_str_dims_t ref_dims;
    map_str_str template_var_values; // str->str templates+values to pass directly to generated code (e.g. lrn params)
    bool operator < ( rtc_func_sig_t const & o ) const { 
      if( fn != o.fn ) { return fn < o.fn; }
      if( ref_dims != o.ref_dims ) { return ref_dims < o.ref_dims; }
      if( template_var_values != o.template_var_values ) { return template_var_values < o.template_var_values; }
      return 0;
    }
    uint32_t get_u32_tvv( string const & tvn ) const { return lc_str_u32( must_find( template_var_values, tvn ) ); }
    string gen_unused_fn( rtc_func_names_map_t & fns ) const;
  };
  typedef shared_ptr< rtc_func_sig_t > p_rtc_func_sig_t; 

  inline std::ostream & operator << ( std::ostream & out, rtc_func_sig_t const & o ) {
    return out << strprintf( "o.fn=%s o.ref_dims=%s", str(o.fn).c_str(), str(o.ref_dims).c_str() );
  }

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

  void insert_nda_dims_sz( map_str_str & mss, string const & nda_vn, dims_t const & dims, bool const & dims_only ) {
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      must_insert( mss, nda_vn+"_"+dims[i].name+"_dim", str(dims[i].sz) );
      if( !dims_only ) { 
	assert_st( dims[i].has_sz_and_stride_and_name() );
	must_insert( mss, nda_vn+"_"+dims[i].name+"_sz", str(dims[i].stride) );
      }
    }
    if( !dims_only ) { must_insert( mss, nda_vn+"_dims_prod", str(dims.dims_prod()) ); } // also emit dim(0)*stride(0)?
  }

  void insert_nda_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() ) {
    if( ix_expr.empty() ) { ix_expr = ix_vn; } // if no custom ix expression, use ix_vn directly as ix expr
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_sz_and_stride_and_name() );
      string v = (dims[i].stride > 1) ? "("+ix_expr+"/"+str(dims[i].stride)+")" : ix_expr;
      must_insert( mss, ix_vn+"_"+dims[i].name+"_nomod", v );      
      // note: we supress the module on the outermost dim, allowing it to overflow (even though we know it's size and
      // could wrap it. i guess if you want such wrapping, add another outmost dim, and you can set it's size to 1? then
      // the value of that dim becomes 0-if-valid, 1-or-more-if-OOB. but then there's unneeded modulos on the outmost
      // dim value, assuming OOB access is guarded against ... ehh, the current system seems okay.
      if( i ) { 
	uint32_t const dsz = dims[i].sz;
	assert_st( dsz );
	if( dsz > 1 ) { v = "("+v+"%%"+str(dims[i].sz)+")"; }
	else { v = "0"; }
      }
      must_insert( mss, ix_vn+"_"+dims[i].name, v );
    }
    must_insert( mss, ix_vn+"_dims_prod", str(dims.dims_prod()) ); // also emit dim(0)*stride(0)?
  }

  dims_t dims_from_nda_spec( string const & nda_spec ) {
    vect_string const dim_names = split( nda_spec, ':' );
    dims_t arg_dims;
    // for now, assume template: (1) can handle any size for all nda dims (to relax, could allow spec of size,
    // then use that instead of 0/wild for sz below) (2) all dims are static (to relax: unclear what
    // syntax/encoding would be)
    for( vect_string::const_iterator i = dim_names.begin(); i != dim_names.end(); ++i ) { arg_dims.add_dims( *i, 0 ); }
    return arg_dims;
  }
  
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
      for( vect_string::const_iterator i = lines.begin(); i != lines.end(); ++i ) {
	// find magic CUCL comment (if any) and process it
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
	  } else { rt_err( "invalid CUCL directive '"+cd+"'. saw:" + *i ); }
	}
      }
    }

    void check_args( rtc_func_sig_t const & rfs_in, vect_arg_decl_t & flat_arg_decls ) {
      assert_st( flat_arg_decls.empty() );
      string arg_check_error;

      // process/flatten-out multi args
      for( vect_arg_decl_t::const_iterator i = arg_decls.begin(); i != arg_decls.end(); ++i ) {
	if( i->multi_base_vn.empty() ) { flat_arg_decls.push_back( *i ); }
	else {
	  uint32_t const multi_sz = rfs_in.get_u32_tvv( i->multi_base_vn + "_num" );
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
	map_str_dims_t::const_iterator adi = rfs_in.ref_dims.find( i->vn );
	if( adi == rfs_in.ref_dims.end() ) {
	  arg_check_error +=  strprintf( "referenced %s arg '%s' not present in ref_dims; ", 
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
      }
      if( !arg_check_error.empty() ) {
	string arg_err = "RTC template function instantiation argument error: " + template_fn + ": " + arg_check_error + "\n";
	for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
	  arg_err += strprintf( "ARG[%s]:\n", str(i - flat_arg_decls.begin()).c_str() );
	  arg_err += strprintf( "  DECL: %s\n", str(*i).c_str() );
	  map_str_dims_t::const_iterator adi = rfs_in.ref_dims.find( i->vn );
	  if( adi != rfs_in.ref_dims.end() ) { arg_err += strprintf( "  CALL: %s\n", str(adi->second).c_str() ); }
	  else { arg_err += "  CALL: not found in ref dims.\n"; }
	}
	arg_err += "full rfs: " + str(rfs_in) + "\n";
	rt_err( arg_err );
      }
    }

  };

  typedef shared_ptr< rtc_template_t > p_rtc_template_t; 
  typedef map< string, p_rtc_template_t > rtc_template_map_t;

  struct rtc_call_gen_t {
    rtc_func_sig_t rfs;
    p_rtc_template_t rtc_func_template;
    vect_arg_decl_t flat_arg_decls; // copy of rtc_func_template->arg_decls, but with multis flattened out
    map_str_str tf_exprs;
    map_str_dims_t all_ix_dims;
    map_str_str cgs; // code-gen sections

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
    void set( string const &var, string const &val ) { must_insert( tf_exprs, var, val ); }

    dims_t const & get_arg_dims_by_name( string const & arg_vn, string const & err_tag = "TEMPLATE" ) {
      map_str_dims_t::const_iterator i = rfs.ref_dims.find( arg_vn );
      if( i == rfs.ref_dims.end() ) { 
	rt_err( strprintf( "referenced %s arg '%s' not present in ref_dims", err_tag.c_str(), arg_vn.c_str() ) );
      }
      return i->second;
    }
    uint32_t get_u32_tvv( string const & tvn ) { return lc_str_u32( must_find( tf_exprs, tvn ) ); }

    void init( p_rtc_template_t const & rtc_func_template_, 
	       custom_codegen_t * const custom_codegen, rtc_func_sig_t const & rfs_, string const & gen_fn_, string & rtc_prog_str ){
      rtc_func_template = rtc_func_template_;
      gen_fn = gen_fn_;
      has_final_flags_arg = 0;
      rfs = rfs_;
      set( "rtc_func_name", gen_fn );
      for( map_str_str::const_iterator i = rfs.template_var_values.begin(); i != rfs.template_var_values.end(); ++i ) {
	set( i->first, i->second );
      }
      rtc_func_template->check_args( rfs, flat_arg_decls );

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
	insert_nda_ix_exprs( tf_exprs, i->ix_vn, ix_dims );
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
      
      if( custom_codegen ) { custom_codegen->gen_op( this, rtc_func_template->template_fn ); }

      // make these always availible as template vars, since why not?
      set( "tpb", str(tpb) ); // should always be fixed/constant/valid (i.e. gen'd kernels never have dynamic tpb)
      set( "blks", str(blks) ); // may be 0 if # of blocks is dynamic
      set( "warp_sz", str("UNKNOWN") ); // yeah, not the best, but probably not exactly wrong. don't use it for real

      instantiate_template( rtc_prog_str );
    }



    void instantiate_template( string & rtc_prog_str ) {
      // generate arg decl code for multi args
      for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
	if( !i->multi_base_vn.empty() ) { line( i->multi_base_vn + "_decl", "GASQ float const * const " + i->vn + "," ); }
	dims_t const & arg_dims = get_arg_dims_by_name( i->vn );
	bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
	insert_nda_dims_sz( tf_exprs, i->vn, arg_dims, dims_only );
      }

      rtc_prog_str += "// -- codegen begins for '"+rtc_func_template->template_fn+
	"'; template substituion table used (bulk sections ommited): --\n";
      for( map_str_str::const_iterator i = tf_exprs.begin(); i != tf_exprs.end(); ++i ) {
	rtc_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
      }
      for( map_str_str::iterator i = cgs.begin(); i != cgs.end(); ++i ) { // terminate and emit bulk cgs
	i->second += "    // end "+i->first;
	set( i->first, i->second ); 
      } 
      lexp_name_val_map_t tf_nvm{ p_lexp_t() };
      tf_nvm.insert_leafs_from( tf_exprs );
      string rtc_func_str;
      try {
	str_format_from_nvm( rtc_func_str, *rtc_func_template->template_str, tf_nvm );
      } catch( rt_exception const & rte ) {
	printf( "rfs=%s\n", str(rfs).c_str() );
	printf( "tf_nvm=%s\n", str(tf_nvm).c_str() );
	rt_err( strprintf( "instantiation failed; see above; rtc_func_template->template_fn=%s; error was: %s\n", 
			   rtc_func_template->template_fn.c_str(), rte.err_msg.c_str() ) );
      }
      rtc_prog_str += rtc_func_str;
    }

  };

  string rtc_func_sig_t::gen_unused_fn( rtc_func_names_map_t & fns ) const {
    string maybe_fn_base = fn;
    set_string unique_dims;
    for( map_str_dims_t::const_iterator ra = ref_dims.begin(); ra != ref_dims.end(); ++ra ) {
      dims_t const & dims = ra->second;
      for( uint32_t i = 0; i != dims.sz(); ++i ) {
	string const & dn = dims.names(i);
	bool const did_ins = unique_dims.insert( dn ).second;
	if( did_ins ) { maybe_fn_base += "__"+dn+"_"+str(dims.dims(i)); }
      }
    }
    string maybe_fn = maybe_fn_base;
    uint32_t uix = 0;
    while( has( fns, maybe_fn ) ) { ++uix; maybe_fn = maybe_fn_base + "__namegenconflict_" + str(uix); }
    return maybe_fn;
  }

  void rtc_call_gen_t::run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, 
				rcg_func_call_t & rcg_func_call, uint32_t const & flags ) {
    rtc_func_call_t rfc;
    rfc.rtc_func_name = rcg_func_call.rtc_func_name;
    rfc.call_tag = rcg_func_call.call_tag;
    rfc.u32_args = rcg_func_call.u32_args;

    for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
      if( i->io_type == "REF" ) { continue; }
      dims_t const & func_dims = get_arg_dims_by_name( i->vn );
      if( func_dims == dims_t() ) { rfc.in_args.push_back("<NULL>"); continue; } // NULL case -- pass though to rtc
      map_str_str::const_iterator an = rcg_func_call.arg_map.find( i->vn );
      if( an == rcg_func_call.arg_map.end() ) {
	rt_err( "specified "+i->io_type+" arg '"+i->vn+"' not found in arg_map at call time." ); 
      }
      // FIXME/note: we can't/don't check call dims for variable-sized ref_dims. this seems less than ideal.
      // one exampl of such usage is in var_stats.
      if( func_dims.has_sz_and_stride_and_name() ) { 
	dims_t const & call_dims = rtc->get_var_dims_floats( an->second );
	if( call_dims != func_dims ) {
	  rt_err( strprintf( "error: dims mismatch at call time for arg %s: call_dims=%s func_dims=%s\n", 
			     i->vn.c_str(), str(call_dims).c_str(), str(func_dims).c_str() ) );
	}	  
      }
      rfc.in_args.push_back( an->second );
    }
    uint32_t call_blks = blks; // if non-zero, blks is static, and we can check arg sizes
    if( !call_blks ) { // handle dynamic # of blks case
      // FIXME: pretty limited / special cased here
      assert_st( rfc.u32_args.size() > 0 );
      call_blks = u32_ceil_div( rfc.u32_args[0], tpb );
    }
    if( show_rtc_calls ) { 
      printf( "%s( in{%s} inout{%s} out{%s} -- u32{%s} ) tpb=%s call_blks=%s\n", str(rfc.rtc_func_name).c_str(), 
	      str(rfc.in_args).c_str(), str(rfc.inout_args).c_str(), str(rfc.out_args).c_str(), str(rfc.u32_args).c_str(),
	      str(tpb).c_str(), str(call_blks).c_str() );
    }
    rfc.tpb.v = tpb;
    rfc.blks.v = call_blks;
    if( has_final_flags_arg ) { rfc.u32_args.push_back( flags ); }
    rtc->run( rfc );
    rcg_func_call.call_id = rfc.call_id;
    // note: temporary rfc is gone after this
  }

  
  struct rtc_codegen_t {
    rtc_template_map_t rtc_templates;

    rtc_func_names_map_t rtc_func_names_map;
    rtc_func_sigs_map_t rtc_func_sigs_map;
    string rtc_prog_str;

    string gen_func( custom_codegen_t * const cc, rtc_func_sig_t const & rfs ) {
      // first, get template
      p_rtc_template_t & rtc_template = rtc_templates[rfs.fn];
      if( !rtc_template ) { rtc_template.reset( new rtc_template_t ); rtc_template->init( rfs.fn ); }

      p_rtc_call_gen_t & rcg = rtc_func_sigs_map[rfs];
      if( !rcg ) { // need to instatiate function and pick unused name
	rcg.reset( new rtc_call_gen_t );
	string gen_fn = rfs.gen_unused_fn( rtc_func_names_map );
	must_insert( rtc_func_names_map, gen_fn, rcg );
	rcg->init( rtc_template, cc, rfs, gen_fn, rtc_prog_str );
      }    
      return rcg->gen_fn;
    }
    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls,
		  rcg_func_call_t & rfc, uint32_t const & flags ) {    
      must_find( rtc_func_names_map, rfc.rtc_func_name )->run_rfc( rtc, show_rtc_calls, rfc, flags );
    }
  };

}
#endif /* _RTC_FUNC_GEN_H_ */
