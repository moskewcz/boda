// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_func_gen.H"

namespace boda 
{
  void rtc_call_geom_t::maybe_update_for_special_cucl_ixs( string const & ix_vn, dims_t const & ix_dims ) {
    // special cases for index var names
    if( ix_vn == "GLOB_ID_1D" ) { 
      // if GLOB_ID_1D is an index for some arg, assume we want 1 thread per element of that arg, and assume block
      // size doesn't matter, so use a reasonable default if it's not already set.
      if( !tpb ) { tpb = get_default_tpb(); }
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
  // ******** arg_decl_t **********
  void arg_decl_t::set_vn_tn( string const & vn_, string const & tn_ ) {
    vn = vn_;
    tn = get_part_before( tn_, "_multi" );
    multi.v = (tn != tn_); // if we found _multi, we stripped it off, so tn is shorter than tn_
  }
  void arg_decl_t::arg_parse( string const & line ) {
    vect_string const arg_decl = split_ws( strip_ws( strip_ending_chars( get_part_before( line, "//" ), " ,);{" ) ) );
    if( arg_decl.size() < 1 ) { rt_err( "invalid CUCL io var decl; no var name found:" + line ); }
    for( vect_string::const_reverse_iterator i = arg_decl.rbegin()+1; i != arg_decl.rend(); ++i ) {
      if( (*i) == "*" ) { ++loi.v; continue; }
      if( (*i) == "const" ) { continue; } // FIXME: for now, ignore const?
      set_vn_tn( arg_decl.back(), *i ); break;
    }
  }
  uint32_t arg_decl_t::get_multi_sz( op_base_t const & op ) const { return multi.v ? op.get_u32( vn+"_num" ) : 1; }
  string arg_decl_t::get_multi_vn( uint32_t const mix ) const { 
    if( multi.v ) { return vn + "_" + str(mix); }
    assert_st( mix == 0 ); return vn;
  }

  // ****** rtc_template_t ********

  void rtc_template_t::init( string const & template_fn_ ) {
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
    
  void rtc_template_t::init_proc_line( string const & line ) {
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
          cad.arg_parse( line );
          if( cad.tn.empty() ) { rt_err( "invalid CUCL io var decl; no var type found." ); }
          if( !(cad.loi.v <= 1) ) { rt_err( "invalid CUCL io var decl; should be exactly zero or one level-of-indirection/*.");}
          if( (cad.loi.v == 0) && (cad.dyn.v) ) { rt_err( "invalid CUCL io var decl; by-value arguments must not be DYN");}
          if( (cad.loi.v == 0) && (cad.io_type == "REF") ) { rt_err( "invalid CUCL io var decl; REF args must not be by-value (since no value(s) will be passed)"); }

          // special case: if type is CUCL type of this var, then there is no restriction on the type that can be
          // passed for this var. note that the only supported types are basic types and this case; there's no
          // ability to use the type of other vars or some other string template or the like to restrict/set the
          // type.
          if( cad.tn == ("%("+cad.vn+"_tn)") ) { cad.tn = ""; } 
          if( cad.tn == "void" ) { cad.tn = "none"; } // void CUCL args want nda_t's with type "none" (FIXME?)

          // another special case: if the var is named "cucl_arg_info", it's where we stuff the arg info
          if( cad.vn == "cucl_arg_info" ) {
            if( has_cucl_arg_info.v ) { rt_err( "duplicate CUCL ARGINFO declartion. must have exactly 0 or 1 ARGINFO arguments."); }
            has_cucl_arg_info.v = 1;
            if( cad.multi.v ) { rt_err( "invalid CUCL ARGINFO decl; should not be multi." ); }
            if( cad.tn.empty() ) { rt_err( "invalid CUCL ARGINFO decl; no var type found." ); }
            if( cad.tn != "%(rtc_func_name)_arg_info_t" ) { rt_err( "invalid CUCL ARGINFO decl; must use '%(rtc_func_name)_arg_info_t' as type name." ); }
            //if( cad.vn != "cucl_arg_info" ) { rt_err( "invalid CUCL ARGINFO decl; must use 'cucl_arg_info' as var name." ); }
            if( cad.loi.v != 0 ) { rt_err( "invalid CUCL ARGINGO decl; should be exactly zero levels-of-indirection/*s:");}  
          }
          for( uint32_t i = 2; i != mmc_parts.size(); ++i ) { 
            cad.ok_dims.push_back( dims_from_nda_spec( cad.tn, mmc_parts[i] ) );
          }
          arg_decls.push_back( cad );
        }
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
  check_args_ret_t rtc_template_t::check_args( op_base_t const & rfs_in ) {
    check_args_ret_t ret;
    // currently, we keep all str_vals. note the the only current ones are type and func_name.
    ret.reduced->str_vals = rfs_in.str_vals; 

    // FIXME: for now, we use a hacky list of 'always keep' val names. maybe these should be in the templates, but there
    // are a couple issues. many/all of these should really be part of the operation (and they are), and really it
    // should be an error if they're *not* used. so we don't want to be listing them in every variant, or if we do, it
    // should only be to check all the proper per-operation-per-varient ones are used. maybe, everything must be used by
    // default? with some way to explicitly ignore some vals that are somehow optional for a given variant?
    vect_string const always_keep_vals{"conv_has_relu"};
    for( vect_string::const_iterator i = always_keep_vals.begin(); i != always_keep_vals.end(); ++i ) {
      if( rfs_in.has( (*i) ) ) { ret.reduced->set( (*i), rfs_in.get( (*i) ) ); }
    }
    string arg_check_error;

    // error check existence of num field for multi args
    for( vect_arg_decl_t::multi_iter i = arg_decls.multi_begin( &rfs_in ); !i.at_end(); ++i ) {
      if( i.vn() == "cucl_arg_info" ) { assert_st( has_cucl_arg_info.v ); continue; } // FIXME: yeah, not great.
      if( i.ad().multi.v ) {
        string const multi_num = i.ad().vn+"_num";
        if( !rfs_in.has(multi_num) ) {
          arg_check_error += strprintf( "mult arg '%s' missing num var '%s' in str_vals; ", 
                                        i.ad().vn.c_str(), multi_num.c_str() );
          i.msz_ = 1; // note: yikes! a bit hacky, we need to skip the entire multi, so fake-set msz=1 and continue.
          continue;
        }
      }
      if( !rfs_in.has(i.vn()) ) {
        arg_check_error +=  strprintf( "referenced %s arg '%s' not present in dims_vals; ", 
                                       i.ad().io_type.c_str(), i.vn().c_str() );
        continue;
      }
      p_nda_t const & arg_nda = rfs_in.get(i.vn());
      dims_t const & arg_dims = arg_nda->dims;
      if( !arg_dims.has_name() ) { arg_check_error += "call arg '"+i.vn()+"' must have names for all dims; "; }
      bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
      if( !dims_only && arg_dims.has_padding() ) { 
        arg_check_error += "call arg '"+i.vn()+"' must not have padding; "; } // FIXME: maybe too strong
      bool matches_decl = 0;
      for( uint32_t j = 0; j != i.ad().ok_dims.size(); ++j ) {
        if( arg_dims.matches_template( i.ad().ok_dims[j] ) ) { matches_decl = 1; break; }
      }
      if( !matches_decl ) { arg_check_error += "call arg '"+str(i.vn())+"' incompatible with decl arg "
          "(dim count mismatch or dim (non-zero and thus req'd) name/size/stride mismatch; "; }
      if( i.ad().loi.v == 0 ) {
        // only scalars are supported for the no-indirection case
        if( arg_dims.dims_prod() != 1 ) { arg_check_error += "call arg '"+str(i.vn())+"' incompatible with decl arg (by-value arguments must be scalar (dims_prod()==1), but call arg had dims_prod()=="+str(arg_dims.dims_prod())+" );  "; }
        assert_st( !i.ad().dyn.v ); 
      } else {
        assert_st( i.ad().loi.v == 1 ); // 'regular' flat nda reference (pointer to block of memory)
      }
      if( i.ad().dyn.v ) {  // if dynamic, zero out sizes/stride, since gen'd func will work for all sizes/strides
        dims_t arg_dims_no_sizes_or_strides = arg_dims;
        arg_dims_no_sizes_or_strides.clear_strides();
        arg_dims_no_sizes_or_strides.clear_dims();
        // FIXME: is this needed/correct: 
        // arg_dims_no_sizes_or_strides.calc_strides(); // for scalars, will set strides_sz = 1 (which is correct enough / better than 0?)
        ret.reduced->set_dims( i.vn(), arg_dims_no_sizes_or_strides );
        // put original nda in to_pass so function will have access to concrete dims (important for DYN REF dims)
        ret.to_pass->set( i.vn(), arg_nda );
      } else {
        ret.reduced->set( i.vn(), arg_nda ); // keep exactly used nda in signature (including value if present)
      }
    }
    if( !arg_check_error.empty() ) {
      string arg_err = "RTC template function instantiation argument error: " + template_fn + ": " + arg_check_error + "\n";
      for( vect_arg_decl_t::multi_iter i = arg_decls.multi_begin( &rfs_in ); !i.at_end(); ++i ) {
        arg_err += strprintf( "ARG[%s]:\n", i.vn().c_str() );
        arg_err += strprintf( "  DECL: %s\n", str(i.ad()).c_str() );
        if( rfs_in.has( i.vn() ) ) { arg_err += strprintf( "  CALL: %s\n", str(rfs_in.get_dims(i.vn())).c_str() ); }
        else { arg_err += "  CALL: not found in ref dims.\n"; }
      }
      arg_err += "full rfs: " + str(rfs_in) + "\n";
      rt_err( arg_err );
    }
    return ret;
  }

  // ****** util funcs ******

  void insert_nda_dims_sz( map_str_str & mss, string const & nda_vn, dims_t const & dims, bool const & dims_only ) {
    assert_st( dims.valid() );
    must_insert( mss, nda_vn+"_tn", str(dims.tn) );
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      must_insert( mss, nda_vn+"_"+dims[i].name+"_dim", str(dims[i].sz) );
      if( !dims_only ) { 
	assert_st( dims[i].has_sz_and_stride_and_name() );
	must_insert( mss, nda_vn+"_"+dims[i].name+"_stride", str(dims[i].stride) );
      }
    }
    if( !dims_only ) { must_insert( mss, nda_vn+"_dims_prod", str(dims.dims_prod()) ); } // also emit dim(0)*stride(0)?
  }

  void insert_nda_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr ) {
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

  string dyn_dims( string const & vn, dims_t const & dims, uint32_t const & dix ) {
    return strprintf( "cucl_arg_info.%s_%s", vn.c_str(), dims[dix].name.c_str() );
  }

  void insert_nda_dyn_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr ) {
    if( ix_expr.empty() ) { ix_expr = ix_vn; } // if no custom ix expression, use ix_vn directly as ix expr
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_name() );
      string v = "("+ix_expr+"/"+dyn_dims(ix_vn,dims,i)+"_stride"+")";
      must_insert( mss, ix_vn+"_"+dims[i].name+"_nomod", v );      
      // note: we supress the module on the outermost dim, allowing it to overflow (even though we know it's size and
      // could wrap it. i guess if you want such wrapping, add another outmost dim, and you can set it's size to 1? then
      // the value of that dim becomes 0-if-valid, 1-or-more-if-OOB. but then there's unneeded modulos on the outmost
      // dim value, assuming OOB access is guarded against ... ehh, the current system seems okay.
      if( i ) { v = "("+v+"%%"+dyn_dims(ix_vn,dims,i)+"_dim"+")"; }
      must_insert( mss, ix_vn+"_"+dims[i].name, v );
    }
    must_insert( mss, ix_vn+"_dims_prod", "cucl_arg_info."+ix_vn+"_dims_prod" ); // also emit dim(0)*stride(0)?
  }

  dims_t dims_from_nda_spec( string const & tn, string const & nda_spec ) {
    dims_t arg_dims;
    if( nda_spec == ":" ) {
      // special case: the string ":" is interpreted as meaning 0 dims (a scalar). otherwise, it would turn into two
      // dims with empty-string as thier name. note that while the empty string is a legal dim name (i.e. no-name, which
      // is a wildcard in the template case), that is currently not supported here (althought perhaps it could be
      // somehow).
    } else {
      vect_string const dim_names = split( nda_spec, ':' );
      // for now, assume template: (1) can handle any size for all nda dims (to relax, could allow spec of size,
      // then use that instead of 0/wild for sz below) (2) all dims are static (to relax: unclear what
      // syntax/encoding would be)
      for( vect_string::const_iterator i = dim_names.begin(); i != dim_names.end(); ++i ) { 
        if( i->empty() ) { rt_err( "invalid (currently forbidden/unused) empty dim name in nda_spec '" + nda_spec +"'" ); }
        arg_dims.add_dims( *i, 0 ); 
      }
    }
    arg_dims.tn = tn; // note: may be empty string for wild type
    arg_dims.calc_strides(1);
    return arg_dims;
  }


  string gen_unused_fn( op_base_t const & op, set_string const & used_names ) {
    string maybe_fn_base = op.get_func_name();
    set_string unique_dims;
    for( map_str_p_nda_t::const_iterator ra = op.nda_vals.begin(); ra != op.nda_vals.end(); ++ra ) {
      dims_t const & dims = ra->second->dims;
      if( !dims.sz() ) { // for scalars, put the name of the scalar and it's value (if it has one)
        maybe_fn_base += "__"+ra->first; 
        if( ra->second->rp_elems() ) { maybe_fn_base += "_"+as_pyid_fixme(get_scalar_c_const_str(*ra->second)); }
      } else {
        for( uint32_t i = 0; i != dims.sz(); ++i ) {
          string const & dn = dims.names(i);
          bool const did_ins = unique_dims.insert( dn ).second;
          if( did_ins ) { maybe_fn_base += "__"+dn+"_"+str(dims.dims(i)); }
        }
      }
    }
    // not the best/most-robust idea, but for now we can avoid most namegenconflicts by (questionally) stuffing the
    // str_vals into the function name as well. it'll be fine, right? this could be removed if problematic.
    for( map_str_str::const_iterator ra = op.str_vals.begin(); ra != op.str_vals.end(); ++ra ) {
      if( ra->first == "func_name" ) { continue; } // must exists, and already used as maybe_fn_base
      if( ra->first == "type" ) { continue; } // usually, the func_name implies the type, so we omit it here
      maybe_fn_base += "__"+ra->first+"_"+as_pyid_fixme(ra->second);
    }

    string maybe_fn = maybe_fn_base;
    uint32_t uix = 0;
    while( has( used_names, maybe_fn ) ) { ++uix; maybe_fn = maybe_fn_base + "__namegenconflict_" + str(uix); }
    return maybe_fn;
  }

  // ****** rtc_call_gen_t ******

  void rtc_call_gen_t::line( string const & sn, string const & line ) { 
    string & cg = cgs[sn];
    if(cg.empty()) { cg += "// begin "+sn+"\n"; } // adding first line in (new) section, add header
    cg += "   " + line + "\n"; 
  }
  void rtc_call_gen_t::set( string const &var, string const &val ) { must_insert( tsvs, var, val ); }
  dims_t const & rtc_call_gen_t::get_arg_dims_by_name( string const & arg_vn, string const & err_tag ) {
    if( !op.has( arg_vn ) ) { 
      rt_err( strprintf( "referenced %s arg '%s' not present in dims_vals", err_tag.c_str(), arg_vn.c_str() ) );
    }
    return op.get_dims( arg_vn );
  }

  // since a CUCL IX may use only a subset of the dims of the arg_dims that it references, we need to create the
  // sub-dims here.
  dims_t apply_use_dims( dims_t const & ix_arg_dims, vect_string const & use_dims ) {
    dims_t ix_dims;
    if( use_dims.empty() ) { ix_dims = ix_arg_dims; } 
    else {
      ix_dims.tn = ix_arg_dims.tn;
      for( vect_string::const_iterator j = use_dims.begin(); j != use_dims.end(); ++j ) {
        dim_t const * use_dim = ix_arg_dims.get_dim_by_name( *j );
        if( !use_dim ) { rt_err( "specified use_dim '"+*j+"' not found in target arg's dims" ); }
        ix_dims.push_back( *use_dim );
      }
      ix_dims.calc_strides(); // note: stride are garbage prior to this call (which is okay)
    }
    return ix_dims;
  }

  void rtc_call_gen_t::init( op_base_t const & op_, p_rtc_template_t const & rtc_func_template_, custom_codegen_t * const cc, string const & gen_fn_ ) {
    op = op_;
    assert_st( gen_fn.empty() ); // double init guard
    gen_fn = gen_fn_;
    assert_st( !gen_fn.empty() ); // so above guard is sound (and a sensible assert anyway)
    set( "rtc_func_name", gen_fn );
    rtc_func_template = rtc_func_template_;
    assert_st( rtc_func_template->template_fn == op.get_func_name() ); // better be right template
    //printf( "op.func_name=%s gen_fn=%s op.str_vals=%s\n", str(op.get_func_name()).c_str(), gen_fn.c_str(), str(op.str_vals).c_str() );
    // if we have a str_val with the magic name 'tpb', use it to set the call geom:
    if( op.has( "tpb" ) ) { rtc_call_geom.tpb = op.get_u32( "tpb" ); } // note: the tpb scalar must have a value for now
    for( vect_ix_decl_t::const_iterator i = rtc_func_template->ix_decls.begin(); i != rtc_func_template->ix_decls.end(); ++i ) {
      dims_t ix_dims = apply_use_dims( get_arg_dims_by_name( i->arg_vn, "IX" ), i->use_dims );
      assert_st( ix_dims.size() && ix_dims.has_name() );
      bool ix_is_dyn = !ix_dims.has_sz_and_stride_and_name();
      if( ix_is_dyn ) { 
        assert_st( i->ix_vn != i->arg_vn ); // FIXME: too strong? but used later to see if this dyn is an ix ...
        dyn_vars.push_back( dyn_dim_info_t{ i->ix_vn, i->arg_vn, i->use_dims } ); 
        add_dyn_nda_dims_sz( i->ix_vn, ix_dims, 0 ); // omit ref templates for used-for-ix-only dyn dims
        insert_nda_dyn_ix_exprs( tsvs, i->ix_vn, ix_dims );
        //rt_err( "NEVER_SAY_NEVER, but can't create CUCL IX for dynamically-sized var" );
      } else {
        ix_dims.calc_strides(); // note: stride are garbage prior to this call (which is okay)
        must_insert( all_ix_dims, i->ix_vn, ix_dims );
        insert_nda_ix_exprs( tsvs, i->ix_vn, ix_dims );
        rtc_call_geom.maybe_update_for_special_cucl_ixs( i->ix_vn, ix_dims );
      }          
    }
    // note: rtc_call_geom is somewhere between unset and fully set here -- dyn cucl ix's may set it more at call
    // time, and failing that there is still the legacy dynamic-blocks case. note, for now we still insist that tpb be
    // set at this point if there are no dyn IXs, and use a default if it was non set by a non-dynamic cucl IX. we may
    // change/relax this later.
#if 0
    if( dyn_vars.empty() ) {
      if( !rtc_call_geom.tpb ) { rtc_call_geom.tpb = rtc_call_geom_t::get_default_tpb(); } // if unset, use a default value
    }
#endif
    // assert_st( rf->blks ); // too strong. if not set, dynamic # blks case

    if( cc ) { cc->gen_op( this, rtc_func_template->template_fn ); } // call custom_codegen_t hook.

    for( vect_arg_decl_t::multi_iter i = rtc_func_template->arg_decls.multi_begin( &op ); !i.at_end(); ++i ) {
      arg_names.push_back( i.vn() ); // in-order-list of function arg names for rtc level
      if( i.vn() == "cucl_arg_info" ) { assert_st( rtc_func_template->has_cucl_arg_info.v ); continue; } // FIXME: yeah, not great.
      if( i.ad().multi.v ) { line( i.ad().vn + "_decl", "GASQ "+i.ad().tn+" const * const "+i.vn()+"," ); }
      p_nda_t const & arg_nda = op.get(i.vn()); // can this fail? if so, need get_arg_dims_by_name()-like error reporting?
      dims_t const & arg_dims = arg_nda->dims;
      if( i.ad().dyn.v ) {
        assert_st( i.ad().loi.v == 1 );
        dyn_vars.push_back( dyn_dim_info_t{ i.vn(), i.vn(), vect_string{} } ); 
        add_dyn_nda_dims_sz( i.vn(), arg_dims, 1 ); 
      } else {	
        bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
        insert_nda_dims_sz( tsvs, i.vn(), arg_dims, dims_only ); 
        if( i.ad().loi.v == 0 ) { // for scalars, set a template varaible that references the scalar (same name as the arg)
          assert_st( arg_dims.dims_prod() == 1 ); // earlier restriction/error-check should guarentee this
          // if we have a constant value now, template can be string constant. otherwise, reference (per-call-dynamic) arg
          set( i.vn(), arg_nda->rp_elems() ? get_scalar_c_const_str(*arg_nda) : i.vn() );
          // printf( "i.vn()=%s tsvs[i.vn()]=%s\n", str(i.vn()).c_str(), str(tsvs[i.vn()]).c_str() );
        } 
      }
    }

    // make these always availible as template vars, since why not?  FIXME: should set these to fields inside
    // cucl_arg_info if they are dynamic. however, codegen that uses tpb directly would still be broken in that case
    // (and such code should probably assert that tpb is non-zero/set). some code might be updatable to use the template
    // string ( which might point to a dynamic value ) instead. NOTE: currently, tpb should always be
    // fixed/constant/valid if there are no dyn vars. set is guarded since a template may declair tpb as a scalar arg
    // (e.g. var_stats).
    if( !has(tsvs,"tpb") ) { set( "tpb", str(rtc_call_geom.tpb) ); } 
    set( "blks", str(rtc_call_geom.blks) ); // may be 0 if # of blocks is dynamic
    set( "warp_sz", str("UNKNOWN") ); // yeah, not the best, but probably not exactly wrong. don't use it for real

    instantiate_template( *rtc_func_template->template_str );
  }

  void rtc_call_gen_t::add_dyn_nda_dims_sz( string const & nda_vn, dims_t const & dims, bool const add_ref_templates ) {
    // FIXME: since it is dynamic, we don't seem to know if nda_vn will be 'dims only' or not here. so for now we
    // assume all dynamic vars are not dims only; this means we'll always generate stride vars for them in the arg
    // info, but maybe that's okay?
    bool const dims_only = 0;
    assert_st( dims.valid() );
    // note: type is still non-dynamic, so we still set it here, similarly to the non-dyn case
    set( nda_vn+"_tn", str(dims.tn) ); 
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      string const vn_dim = nda_vn+"_"+dims[i].name+"_dim";
      line( "cucl_arg_info_decls", cai_tn() + " " + vn_dim + ";" );
      if( add_ref_templates ) { set( vn_dim, "cucl_arg_info."+vn_dim ); }
      if( !dims_only ) { 
        //assert_st( dims[i].has_sz_and_stride_and_name() );
        string const vn_stride = nda_vn+"_"+dims[i].name+"_stride";
        line( "cucl_arg_info_decls", cai_tn() + " " + vn_stride + ";" );
        if( add_ref_templates ) { set( vn_stride, "cucl_arg_info."+vn_stride ); }
      }
    }
    if( !dims_only ) { 
      string const vn_dims_prod = nda_vn+"_dims_prod";
      line( "cucl_arg_info_decls", cai_tn() + " " + vn_dims_prod + ";" );
      if( add_ref_templates ) { set( vn_dims_prod, "cucl_arg_info."+vn_dims_prod ); }
    }
  }

  void add_arg_info_for_dims( dims_t const & dims, vect_int32_t & cucl_arg_info ) {
    // FIXME: since it is dynamic, we don't seem to know if nda_vn will be 'dims only' or not here. so for now we
    // assume all dynamic vars are not dims only; this means we'll always generate stride vars for them in the arg
    // info, but maybe that's okay?
    bool const dims_only = 0;
    assert_st( dims.valid() );
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      cucl_arg_info.push_back( dims[i].sz );
      if( !dims_only ) { cucl_arg_info.push_back( dims[i].stride ); }
    }
    if( !dims_only ) { cucl_arg_info.push_back( dims.dims_prod() ); }
  }

  void rtc_call_gen_t::instantiate_template( string const & template_str ) {
    assert_st( rtc_prog_str.empty() ); // should only call only

    if( (!rtc_func_template->has_cucl_arg_info.v) && (!dyn_vars.empty()) ) {
      rt_err( "template declares _DYN arguments, but no CUCL ARGINFO declaration was found." );
    }
      
    rtc_prog_str += "// -- codegen begins for '"+rtc_func_template->template_fn+
      "'; template substituion table used (bulk sections ommited): --\n";
    for( map_str_str::const_iterator i = tsvs.begin(); i != tsvs.end(); ++i ) {
      rtc_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
    }
    for( map_str_str::iterator i = cgs.begin(); i != cgs.end(); ++i ) { // terminate and emit bulk cgs
      i->second += "    // end "+i->first+"\n";
      set( i->first, i->second ); 
    } 
    lexp_name_val_map_t tf_nvm{ p_lexp_t() };
    tf_nvm.insert_leafs_from( tsvs );
    string rtc_func_str;
    try {
      str_format_from_nvm( rtc_func_str, template_str, tf_nvm );
    } catch( rt_exception const & rte ) {
      printf( "rfs=%s\n", str(op).c_str() );
      rt_err( strprintf( "instantiation failed; see above; type=%s; error was: %s\n", 
                         rtc_func_template->template_fn.c_str(), rte.err_msg.c_str() ) );
    }
    rtc_prog_str += rtc_func_str;      
  }
  
  uint32_t rcg_func_call_t::run( p_rtc_compute_t const & rtc, bool const & show_rtc_calls ) const {
    rtc_func_call_t rfc;
    rfc.rtc_func_name = rcg->gen_fn;
    rtc_call_geom_t dyn_rtc_call_geom = rcg->rtc_call_geom;
    //printf( "op=%s arg_map=%s\n", str(op).c_str(), str(rcg_func_call.arg_map).c_str() );
    p_nda_t cucl_arg_info_nda;
    assert_st( rcg->rtc_func_template->has_cucl_arg_info.v || rcg->dyn_vars.empty() );
    if( rcg->rtc_func_template->has_cucl_arg_info.v ) {
      vect_int32_t cucl_arg_info;
      for( vect_dyn_dim_info_t::const_iterator i = rcg->dyn_vars.begin(); i != rcg->dyn_vars.end(); ++i ) {
        //printf( "i->nda_vn=%s i->src_vn=%s i->use_dims=%s\n", str(i->nda_vn).c_str(), str(i->src_vn).c_str(), str(i->use_dims).c_str() );
        map_str_rtc_arg_t::const_iterator an = arg_map.find( i->src_vn );
        if( an == arg_map.end() ) { rt_err( "<internal error> src arg for dynamic size info '"+i->src_vn+"' not found in arg_map at call time." ); }
        dims_t const & arg_call_dims = an->second.get_dims( *rtc );
        dims_t dyn_call_dims = apply_use_dims( arg_call_dims, i->use_dims );
        //printf( "dyn_call_dims=%s\n", str(dyn_call_dims).c_str() );
        add_arg_info_for_dims( dyn_call_dims, cucl_arg_info );
        if( i->nda_vn != i->src_vn ) { // see earlier FIXME, but for now we use this cond to select IX-derived dyn dims
          dyn_rtc_call_geom.maybe_update_for_special_cucl_ixs( i->nda_vn, dyn_call_dims );
        }
      }
      cucl_arg_info_nda = make_vector_nda( cucl_arg_info ); // convert cucl_arg_info to nda
    }

    for( vect_arg_decl_t::const_iterator i = rcg->rtc_func_template->arg_decls.begin(); 
         i != rcg->rtc_func_template->arg_decls.end(); ++i ) 
    {
      if( i->vn == "cucl_arg_info" ) { // FIXME: not-too-nice special case for cucl_arg_info argument 
        assert_st( rcg->rtc_func_template->has_cucl_arg_info.v );
        must_insert( rfc.arg_map, i->vn, rtc_arg_t{cucl_arg_info_nda} );
        cucl_arg_info_nda.reset(); // mark as used
        continue;
      }
      uint32_t const multi_sz = i->get_multi_sz( rcg->op );
      for( uint32_t mix = 0; mix != multi_sz; ++mix ) {
        string const vn = i->get_multi_vn(mix);
        p_nda_t const & func_nda = rcg->op.get(vn); // can this fail? if so, need get_arg_dims_by_name()-like error reporting?
        dims_t const & func_dims = func_nda->dims;
        dims_t call_dims;
        string call_vn; // for error message
        if( i->loi.v == 0 ) { // arg is some struct/type to be passed by-value 
          if( func_nda->rp_elems() ) { 
            // if in our op, we hard-coded the string constant into the source already, but pass it in dynamically anyway.
            call_dims = func_dims; // note: no mismatch possible
            call_vn = "<internal-error-on-gen-time-constant-arg>";
            must_insert( rfc.arg_map, vn, rtc_arg_t{func_nda} );
          } else { 
            // otherwise, assume the value we want/need is in args
            map_str_rtc_arg_t::const_iterator an = arg_map.find( vn );
            if( an == arg_map.end() ) {
              rt_err( "specified "+i->io_type+" scalar by-value arg '"+vn+"' not found in arg_map at call time." ); 
            }
            call_dims = an->second.get_dims( *rtc );
            call_vn = "<by-value-scalar>";
            if( !an->second.is_nda() ) { rt_err( "UNSUPPORTED: attempted to pass scalar var by-value for arg "+vn ); }
            // either way, pass it to the kernel -- 
            must_insert( rfc.arg_map, vn, rtc_arg_t{an->second} ); // FIXME: should be guarded by below error check
          }
          // FIXME: it's perhaps confusing that the value is always passed dynamically as an arg, even if it's was
          // availible as a constant at gen-time, but it's unclear what else to do here (we need to pass some arg)
          // ... it's probably okay?
        } else { 
          assert_st( i->loi.v == 1 ); // arg is a pointer/reference-to-memory-block
          bool const need_value = ( i->io_type != "REF" ) || ( i->dyn.v ); // if non-REF or DYN-REF, we need an arg
          map_str_rtc_arg_t::const_iterator an = arg_map.find( vn );
          if( need_value && (an == arg_map.end()) ) {
            rt_err( "specified "+i->io_type+" arg '"+vn+"' not found in arg_map at call time." ); 
          }
          // for non-DYN REF args, we allow pulling the call_dims from the op (so call_dims will == func_dims for sure)
          call_dims = (an == arg_map.end()) ? func_dims : an->second.get_dims( *rtc );
          if( ( i->io_type != "REF" ) && (!an->second.is_var()) ) { 
            if( !an->second.v->rp_elems() ) { } // NULL case is handled by by-value processing, allow (FIXME?)
            else { rt_err( "UNSUPPORTED: attempted to pass scalar val by-ref for arg "+vn ); }
          }
          call_vn = (an == arg_map.end()) ? "<from-op>" : ( an->second.is_var() ? an->second.n : "<by-value-nda>" );
          // FIXME: inserts into rfc.arg_map should be guarded by below error check (but don't matter if it fails later)
          if( i->io_type != "REF" ) { must_insert( rfc.arg_map, vn, rtc_arg_t{an->second} );  } 
          else { must_insert( rfc.arg_map, vn, rtc_arg_t{make_dims_nda(call_dims)} ); } // strip data for REF case
        }
        // check that the passed vars are the expected sizes. for non-dyn vars, the sizes must be fully specificed (no
        // wildcards) and be exactly equal. for dyn vars, in particular at least the # of dims per var better match as the
        // cucl_arg_info code assumes this (but here we'll check the dim names too).
        if( !call_dims.matches_template( func_dims ) ) { rt_err( 
            strprintf ("error: dims mismatch at call time. arg=%s: func_dims=%s call_dims=%s call_vn=%s\n", 
                       vn.c_str(), str(func_dims).c_str(), str(call_dims).c_str(), call_vn.c_str()));
        }	  
      }
    }
    assert_st( !bool(cucl_arg_info_nda) ); // if created, should have been used.

    if( show_rtc_calls ) { 
      printf( "%s( %s ) tpb=%s call_blks=%s\n", str(rfc.rtc_func_name).c_str(), 
	      str(rfc.arg_map).c_str(),
	      str(dyn_rtc_call_geom.tpb).c_str(), str(dyn_rtc_call_geom.blks).c_str() );
      //if( !rfc.cucl_arg_info.empty() ) { printf( "  rfc.cucl_arg_info=%s\n", str(rfc.cucl_arg_info).c_str() ); }
    } 
    rfc.tpb.v = dyn_rtc_call_geom.tpb;
    rfc.blks.v = dyn_rtc_call_geom.blks;
    uint32_t const call_id = rtc->run( rfc );
    return call_id;
    // note: temporary rfc is gone after this
  }

  // ****** rtc_codegen_t ******

  using boost::filesystem::is_regular_file;

  p_rcg_func_call_t rtc_codegen_t::gen_func( op_base_t const & rfs_full, map_str_rtc_arg_t const & arg_map ) { // note: no ref to rfs_full is kept
    // first, get template
    string const & func_name = rfs_full.get_func_name();
    p_rtc_template_t & rtc_template = rtc_templates[func_name];
    if( !rtc_template ) { rtc_template.reset( new rtc_template_t ); rtc_template->init( func_name ); }

    check_args_ret_t const ca_ret = rtc_template->check_args( rfs_full );
    // currently, there's nowhere to put any str_vals we're supposed to pass back to the generation function, and it's
    // not clear exactly what that would mean or what the use case would be. but like the whole 'what to do with
    // str_vals' issue, maybe later it'll make sense. shoving them in as 'str' args might be right? but it seems a bit
    // nuts to overload var-ref-string with some-random-strings ...
    if( !ca_ret.to_pass->str_vals.empty() ) { rt_err( "UNIMPLEMENTED: str_vals in ca_ret->to_pass"); }

    p_rtc_call_gen_t & rcg = rtc_func_sigs_map[*ca_ret.reduced];
    if( !rcg ) { // need to instatiate function and pick unused name
      rcg.reset( new rtc_call_gen_t );
      string gen_fn = gen_unused_fn( *ca_ret.reduced, used_names );
      rcg->init( *ca_ret.reduced, rtc_template, cc.get(), gen_fn );
      used_names.insert( gen_fn );
      compile_pend.push_back( rcg );
    } 
    map_str_rtc_arg_t fin_arg_map = arg_map;
    for( map_str_p_nda_t::const_iterator i = ca_ret.to_pass->nda_vals.begin(); i != ca_ret.to_pass->nda_vals.end(); ++i ) {
      if( !has( fin_arg_map, i->first ) ) { // any existing arg_map entry overrides something kept from the op
        must_insert( fin_arg_map, i->first, i->second ); // merge to_pass into arg_map
      }
      // note that if callers with to add to arg_map later, they must bear in mind it may-or-may-not contain any of the
      // pre-existing info from gen-time.
    }
    //printf( "ca_ret.reduced=%s fin_arg_map=%s\n", str(ca_ret.reduced).c_str(), str(fin_arg_map).c_str() );
    return make_shared<rcg_func_call_t>(rcg,fin_arg_map);
  }

  // currently used for calling xpose functions. not the pretiest thing, but better factored out here then dupe'd.
  p_rcg_func_call_t rtc_codegen_t::gen_func_override_func_name( string const & func_name, op_base_t & op, map_str_rtc_arg_t const & arg_map ) {
    // sigh. used only in conv case, where we know func_name is already set ...
    string const orig_func_name = op.get_func_name(); 
    op.erase_func_name();
    op.set_func_name( func_name );
    p_rcg_func_call_t func = gen_func( op, arg_map ); // note: no ref to op better be kept! (and it isn't)
    op.erase_func_name();
    op.set_func_name( orig_func_name );
    return func;
  }

  // compile pending (generated but not compiled) functions 
  void rtc_codegen_t::compile( void ) {
    // nothing pending? do nothing. note that running code below is correct, but calls down to rtc->compile() with no
    // functions, which is probably not the best idea.
    if( compile_pend.empty() ) { return; } 
    vect_rtc_func_info_t rtc_prog_infos;
    for( vect_p_rtc_call_gen_t::const_iterator i = compile_pend.begin(); i != compile_pend.end(); ++i ) {
      rtc_prog_infos.push_back( {(*i)->gen_fn,(*i)->rtc_prog_str,(*i)->arg_names,(*i)->op} );
      (*i)->is_compiled.v = 1; // well, or at least we're going to *try* to compile this func anyway ...
    }
    //printf( "rtc_prog_infos=%s\n", str(rtc_prog_infos).c_str() );
    rtc->compile( rtc_prog_infos, rtc_compile_opts );
    compile_pend.clear();
  }

  uint32_t rtc_codegen_t::run_func( rcg_func_call_t const & call ) {
    compile(); // compile any pending funcs
    return call.run( rtc, rtc_compile_opts.show_rtc_calls );
  }

  // clear functions that aren't externally referenced
  void rtc_codegen_t::clear( void ) {
    // note: this process is a bit tricky. need to be careful about order of operations and maintaining state in all cases
    compile_pend.clear(); // will be recreated on the fly, but need to clear refs temp. for unique ref check to be correct
    for( rtc_func_sigs_map_t::const_iterator i = rtc_func_sigs_map.begin(); i != rtc_func_sigs_map.end(); /*no inc*/ ) {
      if( i->second.unique() ) { // not externally referenced? then remove func.
        if( i->second->is_compiled.v ) { rtc->release_func( i->second->gen_fn ); } // only release from rtc if compiled
        must_erase( used_names, i->second->gen_fn ); // name no longer in use
        i = rtc_func_sigs_map.erase(i); // remove from map
      } else { // ... else, keep func.
        if( !i->second->is_compiled.v ) { compile_pend.push_back( i->second ); } // if pending compiling, re-track
        ++i; // keep func in map
      }
    }
  }
  
  void rtc_codegen_t::gc_clear( void ) { if( rtc_func_sigs_map.size() > 1000 ) { clear(); } }

  void rtc_codegen_t::read_rtc_func_sigs( filename_t const & rtc_func_sigs_fn ) {
    p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      gen_func( *v, map_str_rtc_arg_t() ); // FIXME: this seem broken now; untested/unused? remove? fix?
      uint32_t const ix = i - in_lines->begin();
      if( !(ix % 100000)) { printf( "ix=%s\n", str(ix).c_str() ); }
    }
  }

  void rtc_codegen_t::write_rtc_func_sigs( filename_t const & rtc_func_sigs_fn ) {
    set_op_base_t all_sigs;
    if( is_regular_file( rtc_func_sigs_fn.exp ) ) {  // read in existing contents of file if it exists
      p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
      for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
	p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
	all_sigs.insert( *v );
      }
    }
    // insert func sigs from current codegen set
    for( rtc_func_sigs_map_t::const_iterator i = rtc_func_sigs_map.begin(); i != rtc_func_sigs_map.end(); ++i ) { 
      all_sigs.insert( i->first );
    }
    // write set back out
    p_ostream out = ofs_open( rtc_func_sigs_fn );
    for( set_op_base_t::const_iterator i = all_sigs.begin(); i != all_sigs.end(); ++i ) { (*out) << str( *i ) << "\n"; }
  }

}
