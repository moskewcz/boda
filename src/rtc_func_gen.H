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

  string gen_unused_fn( op_base_t const & op, rtc_func_names_map_t & fns );

  struct arg_decl_t {
    string vn;
    string tn;
    zi_uint32_t loi;
    string io_type;
    zi_bool dyn;
    vect_dims_t ok_dims;
    string multi_base_vn;
  };
  inline std::ostream & operator << ( std::ostream & out, arg_decl_t const & o ) {
    for( vect_dims_t::const_iterator i = o.ok_dims.begin(); i != o.ok_dims.end(); ++i ) {
      out << strprintf( "%s --- vn=%s tn=%s loi=%s io_type=%s ", str(*i).c_str(), str(o.vn).c_str(), 
                        str(o.tn).c_str(), str(o.loi.v).c_str(), str(o.io_type).c_str() );
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
  void insert_nda_dyn_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() );

  dims_t dims_from_nda_spec( string const & tn, string const & nda_spec );

  inline void cucl_ref_decl_parse( arg_decl_t & cad, string const & line ) {
    vect_string const arg_decl = split_ws( strip_ws( strip_ending_chars( get_part_before( line, "//" ), " */" ) ) );
    if( arg_decl.size() < 1 ) { rt_err( "invalid CUCL REF decl; no var name present:" + line ); }
    cad.vn = arg_decl.back();
  }
  inline void cucl_arg_decl_parse( arg_decl_t & cad, string const & line ) {
    vect_string const arg_decl = split_ws( strip_ws( strip_ending_chars( get_part_before( line, "//" ), " ,);{" ) ) );
    if( arg_decl.size() < 1 ) { rt_err( "invalid CUCL io var decl; no var name found:" + line ); }
    cad.vn = arg_decl.back();
    for( vect_string::const_reverse_iterator i = arg_decl.rbegin()+1; i != arg_decl.rend(); ++i ) {
      if( (*i) == "*" ) { ++cad.loi.v; continue; }
      if( (*i) == "const" ) { continue; } // FIXME: for now, ignore const?
      cad.tn = *i; break;
    }
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
        bool const multi = maybe_strip_suffix( cd, "_MULTI" );
        if( (cd == "IN") || (cd == "INOUT") || (cd == "OUT") || (cd == "REF") || (cd == "IX") ) { 
          if( cd == "IX" ) {
            if( dyn ) { rt_err( "invalid use of _DYN suffix on CUCL IX decl" ); }
            if( multi ) { rt_err( "invalid use of _MULTI suffix on CUCL IX decl" ); }
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
              cucl_ref_decl_parse( cad, line );
            } else {
              cucl_arg_decl_parse( cad, line );
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
            if( multi ) {
              cad.multi_base_vn = cad.vn;
              bool const did_strip = maybe_strip_suffix( cad.multi_base_vn, "_N" );
              if( !did_strip ) { rt_err("MULTI arg names must end with _N (to be _0, _1, ..."); }
            }
            arg_decls.push_back( cad );
          }
        } else if( cd == "ARGINFO" ) {
          if( has_cucl_arg_info.v ) { rt_err( "duplicate CUCL ARGINFO declartion. must have exactly 0 or 1 ARGINFO arguments."); }
          has_cucl_arg_info.v = 1;
          arg_decl_t cad;
          cucl_arg_decl_parse( cad, line );
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
	  if( arg_dims == dims_t() ) { matches_decl = 1; break; } // NULL case
	  if( arg_dims.matches_template( i->ok_dims[j] ) ) { matches_decl = 1; break; }
	}
	if( !matches_decl ) { arg_check_error += "call arg "+str(ix)+" incompatible with decl arg "
	    "(dim count mismatch or dim (non-zero and thus req'd) name/size/stride mismatch; "; }
        if( i->dyn.v ) {  // if dynamic, zero out sizes/stride, since gen'd func will work for all sizes/strides
          dims_t arg_dims_no_sizes_or_strides = arg_dims;
          arg_dims_no_sizes_or_strides.clear_strides();
          arg_dims_no_sizes_or_strides.clear_dims();
          must_insert( rfs_out->dims_vals, i->vn, arg_dims_no_sizes_or_strides ); 
        } else {
          must_insert( rfs_out->dims_vals, i->vn, arg_dims ); // keep exactly used dims_vals in signature
        }
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
    vect_arg_decl_t flat_arg_decls; // copy of rtc_func_template->arg_decls, but with multis flattened out
    map_str_dims_t all_ix_dims;
    map_str_str cgs; // code-gen sections
    string rtc_prog_str; // final full program string
    
    // info needed to make call
    string gen_fn;
    bool has_final_flags_arg;
    rtc_call_geom_t rtc_call_geom;
    p_rtc_template_t rtc_func_template; // post-init, currently used only for dynamic ix_decl processing at call time
    vect_dyn_dim_info_t dyn_vars;
    void run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, rcg_func_call_t & rfc, uint32_t const & flags);

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

    rtc_call_gen_t( op_base_t const & rfs_ ) : op_base_t( rfs_ ) { }

    void init( p_rtc_template_t const & rtc_func_template_, vect_arg_decl_t const & flat_arg_decls_, 
	       custom_codegen_t * const custom_codegen, string const & gen_fn_ ){
      assert_st( gen_fn.empty() ); // double init guard
      gen_fn = gen_fn_;
      assert_st( !gen_fn.empty() ); // so above guard is sound (and a sensible assert anyway)
      has_final_flags_arg = 0;
      set( "rtc_func_name", gen_fn );
      flat_arg_decls = flat_arg_decls_;
      rtc_func_template = rtc_func_template_;
      // if we have a str_val with the magic name 'tpb', use it to set the call geom:
      if( has( str_vals, "tpb" ) ) { rtc_call_geom.tpb = get_u32( "tpb" ); }
      for( vect_ix_decl_t::const_iterator i = rtc_func_template->ix_decls.begin(); i != rtc_func_template->ix_decls.end(); ++i ) {
	dims_t ix_dims = apply_use_dims( get_arg_dims_by_name( i->arg_vn, "IX" ), i->use_dims );
        assert_st( ix_dims.size() && ix_dims.has_name() );
        bool ix_is_dyn = !ix_dims.has_sz_and_stride_and_name();
	if( ix_is_dyn ) { 
          assert_st( i->ix_vn != i->arg_vn ); // FIXME: too strong? but used later to see if this dyn is an ix ...
          dyn_vars.push_back( dyn_dim_info_t{ i->ix_vn, i->arg_vn, i->use_dims } ); 
          add_dyn_nda_dims_sz( i->ix_vn, ix_dims, 0 ); // omit ref templates for used-for-ix-only dyn dims
          insert_nda_dyn_ix_exprs( str_vals, i->ix_vn, ix_dims );
          //rt_err( "NEVER_SAY_NEVER, but can't create CUCL IX for dynamically-sized var" );
        } else {
          ix_dims.calc_strides(); // note: stride are garbage prior to this call (which is okay)
          must_insert( all_ix_dims, i->ix_vn, ix_dims );
          insert_nda_ix_exprs( str_vals, i->ix_vn, ix_dims );
          rtc_call_geom.maybe_update_for_special_cucl_ixs( i->ix_vn, ix_dims );
        }          
      }
      // note: rtc_call_geom is somewhere between unset and fully set here -- dyn cucl ix's may set it more at call
      // time, and failing that there is still the legacy dynamic-blocks case. note, for now we still insist that tpb be
      // set at this point if there are no dyn IXs, and use a default if it was non set by a non-dynamic cucl IX. we may
      // change/relax this later.
#if 0
      if( dyn_vars.empty() ) {
        if( !rtc_call_geom.tpb ) { rtc_call_geom.tpb = rtc_call_geom_t::default_tpb; } // if unset, use a default value
      }
#endif
      // assert_st( rf->blks ); // too strong. if not set, dynamic # blks case

      if( custom_codegen ) { custom_codegen->gen_op( this, type ); }

      // make these always availible as template vars, since why not?  FIXME: should set these to fields inside
      // cucl_arg_info if they are dynamic. however, codegen that uses tpb directly would still be broken in that case
      // (and such code should probably assert that tpb is non-zero/set). some code might be updatable to use the
      // template string ( which might point to a dynamic value ) instead.
      if( !has( str_vals, "tpb" ) ) { set( "tpb", str(rtc_call_geom.tpb) ); } // currently, should always be fixed/constant/valid if there are no dyn vars
      set( "blks", str(rtc_call_geom.blks) ); // may be 0 if # of blocks is dynamic
      set( "warp_sz", str("UNKNOWN") ); // yeah, not the best, but probably not exactly wrong. don't use it for real

      instantiate_template( *rtc_func_template->template_str );
    }

    string cai_tn( void ) const { return "int32_t"; } // type string for cucl arg info vars
    void add_dyn_nda_dims_sz( string const & nda_vn, dims_t const & dims, bool const add_ref_templates ) {
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

    void instantiate_template( string const & template_str ) {
      assert_st( rtc_prog_str.empty() ); // should only call only
      for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
	if( !i->multi_base_vn.empty() ) { line( i->multi_base_vn + "_decl", "GASQ float const * const " + i->vn + "," ); }
	dims_t const & arg_dims = get_arg_dims_by_name( i->vn );
        if( arg_dims == dims_t() ) { continue; } // skip null dims
        if( i->dyn.v ) { 
          dyn_vars.push_back( dyn_dim_info_t{ i->vn, i->vn, vect_string{} } ); 
          add_dyn_nda_dims_sz( i->vn, arg_dims, 1 ); 
        } else {	
          bool const dims_only = !arg_dims.has_sz_and_stride_and_name();
          insert_nda_dims_sz( str_vals, i->vn, arg_dims, dims_only ); 
        }
      }

      if( (!rtc_func_template->has_cucl_arg_info.v) && (!dyn_vars.empty()) ) {
        rt_err( "template declares _DYN arguments, but no CUCL ARGINFO declaration was found." );
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
