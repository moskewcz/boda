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

  struct custom_codegen_t { virtual void gen_op( rtc_call_gen_t * rcg, string const & op_name ) = 0; };

  struct rtc_func_sig_t { 
    string fn;
    vect_dims_t args;
    map_str_dims_t ref_dims;
    map_str_str template_var_values; // str->str templates+values to pass directly to generated code (e.g. lrn params)
    bool operator < ( rtc_func_sig_t const & o ) const { 
      if( fn != o.fn ) { return fn < o.fn; }
      if( args != o.args ) { return args < o.args; }
      if( ref_dims != o.ref_dims ) { return ref_dims < o.ref_dims; }
      if( template_var_values != o.template_var_values ) { return template_var_values < o.template_var_values; }
      return 0;
    }
    uint32_t get_u32_tvv( string const & tvn ) { return lc_str_u32( must_find( template_var_values, tvn ) ); }
    string gen_unused_fn( rtc_func_names_map_t & fns ) const;
  };
  inline std::ostream & operator << ( std::ostream & out, rtc_func_sig_t const & o ) {
    return out << strprintf( "o.fn=%s o.args=%s", str(o.fn).c_str(), str(o.args).c_str() );
  }

  struct arg_decl_t {
    string vn;
    string io_type;
    vect_dims_t ok_dims;
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

  void insert_nda_dims_sz( vect_pair_str_str & mss, string const & nda_vn, dims_t const & dims ) {
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_sz_and_stride_and_name() );
      mss.push_back( make_pair( nda_vn+"_"+dims[i].name+"_dim", str(dims[i].sz) ) );
      mss.push_back( make_pair( nda_vn+"_"+dims[i].name+"_sz", str(dims[i].stride) ) );
    }
    mss.push_back( make_pair( nda_vn+"_dims_prod", str(dims.dims_prod()) ) ); // also emit dim(0)*stride(0)?
  }

  void insert_nda_ix_exprs( vect_pair_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() ) {
    if( ix_expr.empty() ) { ix_expr = ix_vn; } // if no custom ix expression, use ix_vn directly as ix expr
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_sz_and_stride_and_name() );
      string v = (dims[i].stride > 1) ? "("+ix_expr+"/"+str(dims[i].stride)+")" : ix_expr;
      mss.push_back( make_pair( ix_vn+"_"+dims[i].name+"_nomod", v ) );      
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
      mss.push_back( make_pair( ix_vn+"_"+dims[i].name, v ) );
    }
    mss.push_back( make_pair( ix_vn+"_dims_prod", str(dims.dims_prod()) ) ); // also emit dim(0)*stride(0)?
  }
  
  struct rtc_call_gen_t {
    rtc_func_sig_t rfs;
    p_string rtc_func_template; // read from file
    vect_arg_decl_t arg_decls; // parsed out of rtc_func_template
    vect_ix_decl_t ix_decls; // parsed out of rtc_func_template
    vect_pair_str_str tf_exprs;
    map_str_dims_t all_ix_dims;
    map_str_str cgs; // code-gen sections

    // info needed to make call
    string gen_fn;
    bool has_final_flags_arg;
    vect_uint32_t arg_sizes;
    uint32_t tpb;
    uint32_t blks;

    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, 
		  rtc_func_call_t & rfc, map_str_str const & arg_map, uint32_t const & flags );

    void line( string const & sn, string const & line ) { 
      string & cg = cgs[sn];
      if(cg.empty()) { cg += "// begin "+sn+"\n"; } // adding first line in (new) section, add header
      cg += "   " + line + "\n"; 
    }
    void set( string const &var, string const &val ) { tf_exprs.push_back( std::make_pair( var, val )); }


    dims_t const & get_arg_dims_by_name( string const & arg_vn ) {
      for( uint32_t i = 0; i != arg_decls.size(); ++i ) { if( arg_decls[i].vn == arg_vn ) { return rfs.args.at(i); } }
      rt_err( strprintf( "referenced arg '%s' not declared\n", str(arg_vn).c_str() ) );
    }
    
    void init( custom_codegen_t * const custom_codegen, rtc_func_sig_t const & rfs_, string const & gen_fn_, string & rtc_prog_str ) {
      gen_fn = gen_fn_;
      has_final_flags_arg = 0;
      rfs = rfs_;
      tf_exprs.push_back( make_pair( "rtc_func_name", gen_fn ) );
      for( map_str_str::const_iterator i = rfs.template_var_values.begin(); i != rfs.template_var_values.end(); ++i ) {
	tf_exprs.push_back( make_pair( i->first, i->second ) );
      }
      rtc_func_template = read_whole_fn( (path(py_boda_test_dir()) / "rtc" / (rfs.fn+".cucl")).string() );
      parse_template();
      for( uint32_t i = 0; i != arg_decls.size(); ++i ) {
	if( i < rfs.args.size() ) { continue; } // arg was passed positionally, no need to search by name
	assert_st( rfs.args.size() == i ); // otherwise, we should be finding the next arg
	if( !has( rfs.ref_dims, arg_decls[i].vn ) ) { 
	  rt_err( "specified "+arg_decls[i].io_type+" dim '"+arg_decls[i].vn+
		  "' not found during template instantiation." ); }
	rfs.args.push_back( must_find( rfs.ref_dims, arg_decls[i].vn ) );
      }
      check_args();
      uint32_t num_nonref_args = 0;
      for( uint32_t i = 0; i != rfs.args.size(); ++i ) { 
	if( rfs.args[i].has_sz_and_stride_and_name() ) { insert_nda_dims_sz( tf_exprs, arg_decls[i].vn, rfs.args[i] ); }
	if( arg_decls[i].io_type != "REF" ) { ++num_nonref_args; }
      }

      arg_sizes.resize( num_nonref_args ); // values unused, but size must match # args at call time
      // FIXME: gen_call() doesn't handle num_nonref_args != arg_decls.size() case yet
      tpb = 0;
      blks = 0;
      uint32_t const default_tpb = 256; // FIXME: make 'reasonable default' tpb configurable/heurisitic
      for( vect_ix_decl_t::const_iterator i = ix_decls.begin(); i != ix_decls.end(); ++i ) {
	dims_t const & ix_arg_dims = get_arg_dims_by_name( i->arg_vn );
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
      
      if( custom_codegen ) { custom_codegen->gen_op( this, rfs.fn ); }

      // make these always availible as template vars, since why not?
      tf_exprs.push_back( make_pair( "tpb", str(tpb) ) ); // should always be fixed/constant/valid (i.e. gen'd kernels never have dynamic tpb)
      tf_exprs.push_back( make_pair( "blks", str(blks) ) ); // may be 0 if # of blocks is dynamic
      tf_exprs.push_back( make_pair( "warp_sz", str("UNKNOWN") ) ); // yeah, not the best, but probably not exactly wrong. don't use it for real

      instantiate_template( rtc_prog_str );
    }

    void check_args( void ) {
      string arg_check_error;
      if( rfs.args.size() != arg_decls.size() ) { arg_check_error = "declared/called argument count mismatch"; }
      else {
	for( uint32_t i = 0; i != rfs.args.size(); ++i ) {
	  if( !rfs.args[i].has_name() ) { arg_check_error += "call arg "+str(i)+ " must have names for all dims; "; }
	  if( rfs.args[i].has_sz_and_stride_and_name() && rfs.args[i].has_padding() ) { 
	    arg_check_error += "call args "+str(i)+ " must not have padding; "; } // FIXME: maybe too strong
	  bool matches_decl = 0;
	  for( uint32_t j = 0; j != arg_decls[i].ok_dims.size(); ++j ) {
	    if( rfs.args[i].matches_template( arg_decls[i].ok_dims[j] ) ) { matches_decl = 1; }
	  }
	  if( !matches_decl ) { arg_check_error += "call arg "+str(i)+" incompatible with decl arg "
	      "(dim count mismatch or dim (non-zero and thus req'd) name/size/stride mismatch; "; }
	}
      }
      if( !arg_check_error.empty() ) {
	string arg_err = "RTC template function instantiation argument error: " + rfs.fn + ": " + arg_check_error + "\n";
	for( uint32_t i = 0; i != std::max(rfs.args.size(),arg_decls.size()); ++i ) {
	  arg_err += strprintf( "ARG[%s]:\n", str(i).c_str() );
	  if( i < arg_decls.size() ) { arg_err += strprintf( "  DECL: %s\n", str(arg_decls[i]).c_str() ); }
	  if( i < rfs.args.size() ) {  arg_err += strprintf( "  CALL: %s\n", str(rfs.args[i]).c_str() ); }
	}
	rt_err( arg_err );
      }
    }

    void instantiate_template( string & rtc_prog_str ) {
      rtc_prog_str += "// -- codegen begins for '"+rfs.fn+"'; template substituion table used (bulk sections ommited): --\n";
      for( vect_pair_str_str::const_iterator i = tf_exprs.begin(); i != tf_exprs.end(); ++i ) {
	rtc_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
      }
      for( map_str_str::iterator i = cgs.begin(); i != cgs.end(); ++i ) { // terminate and emit bulk cgs
	i->second += "    // end "+i->first;
	tf_exprs.push_back( make_pair( i->first, i->second ) ); 
      } 
      lexp_name_val_map_t tf_nvm{ p_lexp_t() };
      tf_nvm.insert_leafs_from( tf_exprs );
      string rtc_func_str;
      try {
	str_format_from_nvm( rtc_func_str, *rtc_func_template, tf_nvm );
      } catch( rt_exception const & rte ) {
	printf( "rfs=%s\n", str(rfs).c_str() );
	printf( "tf_nvm=%s\n", str(tf_nvm).c_str() );
	rt_err( strprintf( "instantiation failed; see above; error was: %s\n", rte.err_msg.c_str() ) );
      }
      rtc_prog_str += rtc_func_str;
    }

    static dims_t dims_from_nda_spec( string const & nda_spec ) {
      vect_string const dim_names = split( nda_spec, ':' );
      dims_t arg_dims;
      // for now, assume template: (1) can handle any size for all nda dims (to relax, could allow spec of size,
      // then use that instead of 0/wild for sz below) (2) all dims are static (to relax: unclear what
      // syntax/encoding would be)
      for( vect_string::const_iterator i = dim_names.begin(); i != dim_names.end(); ++i ) { arg_dims.add_dims( *i, 0 ); }
      return arg_dims;
    }

    void parse_template( void ) {
      vect_string lines = split( *rtc_func_template, '\n' );
      for( vect_string::const_iterator i = lines.begin(); i != lines.end(); ++i ) {
	// find magic CUCL comment (if any) and process it
	string const mmc = get_part_after( *i, "//" );
	vect_string mmc_parts = split_ws( strip_ws( mmc ) );
	if( (mmc_parts.size() > 0) && (mmc_parts[0] == "CUCL" ) ) {
	  if( mmc_parts.size() < 2 ) { rt_err( "invalid CUCL magic comment. missing directive after CUCL. saw: " + *i ); }
	  string const & cd = mmc_parts[1];
	  if( (cd == "IN") || (cd == "INOUT") || (cd == "OUT") || (cd == "REF") || (cd == "IX") ) { 
	    if( cd == "IX" ) {
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
	      string const arg_name = arg_decl.back();
	      vect_dims_t ok_dims;
	      for( uint32_t i = 2; i != mmc_parts.size(); ++i ) { ok_dims.push_back( dims_from_nda_spec( mmc_parts[i] ) ); }
	      arg_decls.push_back( arg_decl_t{ arg_name, cd, ok_dims } );
	    }
	  } else { rt_err( "invalid CUCL directive '"+cd+"'. saw:" + *i ); }
	}
      }
    }
  };

  string rtc_func_sig_t::gen_unused_fn( rtc_func_names_map_t & fns ) const {
    string maybe_fn_base = fn;
    set_string unique_dims;
    vect_dims_t all_dims = args;
    for( map_str_dims_t::const_iterator i = ref_dims.begin(); i != ref_dims.end(); ++i ) { all_dims.push_back( i->second ); } // note: name ignores
    for( vect_dims_t::const_iterator i = args.begin(); i != args.end(); ++i ) {
      dims_t const & dims = *i;
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
				rtc_func_call_t & rfc, map_str_str const & arg_map, uint32_t const & flags ) {    
    bool clr = 0;
    if( (rfc.in_args.size()+rfc.inout_args.size()+rfc.out_args.size()) == 0 ) { // FIXME: HACK: gate new flow on rfc args being empty (transitionally)
      clr = 1;
      for( uint32_t i = 0; i != arg_decls.size(); ++i ) {
	if( arg_decls[i].io_type == "REF" ) { continue; }
	map_str_str::const_iterator an = arg_map.find( arg_decls[i].vn );
	if( an == arg_map.end() ) {
	  rt_err( "specified "+arg_decls[i].io_type+" arg '"+arg_decls[i].vn+"' not found in arg_map at call time." ); 
	}
	rfc.in_args.push_back( an->second );
      }
    }

    uint32_t call_blks = blks; // if non-zero, blks is static, and we can check arg sizes
    //printf( "name=%s arg_sizes=%s rfc.args.size()=%s\n", str(name).c_str(), str(arg_sizes).c_str(), str(rfc.args.size()).c_str() );

    // FIXME: arg size checking bot broken/removed during the
    // ocl/nvrtc split/factoring out. oops. also, arg_sizes (or
    // whatever it becomes) probably needs to be split into
    // in/inout/out parts or something.
    if( call_blks ) { assert( arg_sizes.size() == (rfc.in_args.size()+rfc.inout_args.size()+rfc.out_args.size()) ); }
    // FIXME: check that we're passing the correct # of args here somehow.
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
    if( has_final_flags_arg ) { rfc.u32_args.pop_back(); }
    if( clr ) { rfc.in_args.clear(); }
  }

  
  struct rtc_codegen_t {
    rtc_func_names_map_t rtc_func_names_map;
    rtc_func_sigs_map_t rtc_func_sigs_map;
    string rtc_prog_str;

    string gen_func( custom_codegen_t * const cc, rtc_func_sig_t const & rfs ) {
      p_rtc_call_gen_t & rcg = rtc_func_sigs_map[rfs];
      if( !rcg ) { // need to instatiate function and pick unused name
	rcg.reset( new rtc_call_gen_t );
	string gen_fn = rfs.gen_unused_fn( rtc_func_names_map );
	must_insert( rtc_func_names_map, gen_fn, rcg );
	rcg->init( cc, rfs, gen_fn, rtc_prog_str );
      }    
      return rcg->gen_fn;
    }
    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls,
		  rtc_func_call_t & rfc, uint32_t const & flags ) {    
      must_find( rtc_func_names_map, rfc.rtc_func_name )->run_rfc( rtc, show_rtc_calls, rfc, rfc.arg_map, flags );
    }
  };

}
#endif /* _RTC_FUNC_GEN_H_ */
