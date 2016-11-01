// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"op-tuner.H"
#include"str_util.H"
#include"nesi.H"
#include"lexp.H"
#include"has_main.H"
#include<sstream>
#include<boost/regex.hpp>

namespace boda 
{

  using boost::regex;
  using boost::regex_search;

  bool read_string_or_EOF( p_istream const & in, string const & s ) {
    string line;
    if( ifs_getline( "", in, line ) ) { return true; } // EOF
    if( line != s ) { rt_err( strprintf( "error reading line-oriented text stream. expected a line with '%s', but saw '%s'.", 
                                         str(s).c_str(), str(line).c_str() ) ); }
    return false;
  }
  void must_read_string( p_istream const & in, string const & s ) {
    bool const at_eof = read_string_or_EOF( in, s );
    if( at_eof ) { rt_err( strprintf( "error reading line-oriented text stream. expected a line with '%s', but got EOF.", 
                                      str(s).c_str() ) ); }
  }
  string must_getline( p_istream const & in ) {
    string ret;
    bool const at_eof = ifs_getline( "", in, ret );
    if( at_eof ) { rt_err( "error reading line-oriented text stream. expected a non-empty line, but got EOF." ); }
    return ret;
  }

  template< typename T > inline void bread_from_str( string const & in, T & v ) {
    std::istringstream iss( in );
    bread( iss, v );
  }

  void read_op_run_t( op_run_t & v, p_istream const & in ) {
    v.be_plat_tag = must_getline( in );
    v.rt_secs = lc_str_d( must_getline( in ) );
    v.err = must_getline( in );
    if( v.err.empty() ) { v.op = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( must_getline( in ) ), 0 ); }
  }

  void read_op_tune_wisdom( op_tune_wisdom_t & v, p_istream const & in ) {
    v.op_tune = make_p_op_tune_t_init_and_check_unused_from_lexp( parse_lexp( must_getline( in ) ), 0 );     
    string line;
    while( 1 ) {
      line = must_getline( in );
      if( 0 ) {
      } else if( line == "/op_tune_wisdom_t" ) { 
        return;
      } else if( line == "op_run_t" ) {
        op_run_t vv;
        read_op_run_t( vv, in );
        must_insert( v.runs, vv.be_plat_tag, vv );
      } else {
        rt_err( strprintf( "unknown op_tune_wisdom_t text format stream command read '%s'", line.c_str() ) );
      }
    }
  }

  // returns null if no more wisdoms in stream
  p_op_wisdom_t read_next_wisdom( p_istream const & in ) {
    p_op_wisdom_t ret;
    if( read_string_or_EOF( in, "op_wisdom_t" ) ) { return ret; } // EOF
    ret = make_shared<op_wisdom_t>();
    ret->op = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( must_getline( in ) ), 0 ); 
    string line;
    while( 1 ) {
      line = must_getline( in );
      if( 0 ) {
      } else if( line == "/op_wisdom_t" ) { 
        return ret;
      } else if( line == "kg" ) { 
        pair_str_p_nda_digest_t kg;
        kg.first = must_getline( in );
        bread_from_str( unhex(must_getline( in )), kg.second );
        ret->kgs.push_back( kg );
      } else if( line == "op_tune_wisdom_t" ) { 
        p_op_tune_wisdom_t v = make_shared< op_tune_wisdom_t >();
        read_op_tune_wisdom( *v, in );
        ret->wisdoms.push_back( v );
      } else { 
        rt_err( strprintf( "unknown op_wisdom_t text format stream command read '%s'", line.c_str() ) );
      }
    }
    return ret; // may be null if at eof
  }

  void write_op_run( op_run_t const & v, std::ostream & out ) {
    out << "op_run_t\n";
    out << v.be_plat_tag << "\n";
    out << v.rt_secs << "\n";
    out << v.err << "\n";
    if( v.err.empty() ) { assert_st( v.op ); out << str(v.op) << "\n"; }
  }

  void write_op_tune_wisdom( op_tune_wisdom_t const & v, std::ostream & out ) {
    out << "op_tune_wisdom_t\n";
    out << str(v.op_tune) << "\n";
    for( map_str_op_run_t::const_iterator i = v.runs.begin(); i != v.runs.end(); ++i ) { write_op_run( i->second, out ); }
    out << "/op_tune_wisdom_t\n";
  }

  template< typename T > inline string bwrite_to_str( T const & v ) {
    std::ostringstream oss;
    bwrite( oss, v );
    return oss.str();
  }

  void write_op_wisdom( op_wisdom_t const & v, std::ostream & out ) {
    out << "op_wisdom_t\n";
    out << str(v.op) << "\n";
    for( vect_pair_str_p_nda_digest_t::const_iterator i = v.kgs.begin(); i != v.kgs.end(); ++i ) {
      out << "kg\n" << i->first << "\n" << hex(bwrite_to_str(i->second)) << "\n"; 
    }
    for( vect_p_op_tune_wisdom_t::const_iterator i = v.wisdoms.begin(); i != v.wisdoms.end(); ++i ) {
      write_op_tune_wisdom( **i, out );
    }
    out << "/op_wisdom_t\n";
  }

  void op_tune_wisdom_t::merge_runs_from( op_tune_wisdom_t const & o ) {
    // for now, don't allow overwrite ...
    for( map_str_op_run_t::const_iterator i = o.runs.begin(); i != o.runs.end(); ++i ) {
      must_insert( runs, i->first, i->second );
    }
  }

  void filter_runs( op_wisdom_t & owi, regex const & r_plat ) {
    for( vect_p_op_tune_wisdom_t::const_iterator i = owi.wisdoms.begin(); i != owi.wisdoms.end(); ++i ) {
      for( map_str_op_run_t::iterator j = (*i)->runs.begin(); j != (*i)->runs.end(); ) {
        op_run_t const & r = j->second;
        if( (!r.err.empty()) || (!regex_search( r.be_plat_tag, r_plat )) ) { j = (*i)->runs.erase(j); }
        else { ++j; }
      }
    }
  }
      
  void by_op_tune_set_p_op_tune_wisdom_t::add_runs( vect_p_op_tune_wisdom_t const & wisdoms ) {
    for( vect_p_op_tune_wisdom_t::const_iterator i = wisdoms.begin(); i != wisdoms.end(); ++i ) {
      std::pair<iterator,bool> ins_ret = this->insert( *i );
      if( !ins_ret.second ) { (*ins_ret.first)->merge_runs_from( **i ); }
    }
  }

  void op_wisdom_t::merge_wisdoms_from( op_wisdom_t const & o ) {
    by_op_tune_set_p_op_tune_wisdom_t all_otw;
    all_otw.add_runs( wisdoms );
    all_otw.add_runs( o.wisdoms );
    wisdoms.clear();
    for( by_op_tune_set_p_op_tune_wisdom_t::const_iterator i = all_otw.begin(); i != all_otw.end(); ++i ) { wisdoms.push_back( *i ); }
  }

  // merge multiple wisdom files into one. note: output ops will be sorted by op, not by order in any input files.
  struct wis_merge_t : virtual public nesi, public has_main_t // NESI(help="analyses wisdom file, output data in format for plotting",
           // bases=["has_main_t"], type_id="wis-merge" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_filename_t wisdom_in_fns; //NESI(help="wisdom input files to merge")
    filename_t wisdom_out_fn; //NESI(help="output merged wisdom file",req=1)
    uint32_t keep_kgs; //NESI(default="0",help="if zero, kgs are dropped. else, kgs from the first input file with a given op are kept.")
    by_op_set_p_op_wisdom_t all_wis;
    void proc_fn( filename_t const & win_fn ) {
      p_istream win = ifs_open( win_fn );
      for( p_op_wisdom_t owi; owi = read_next_wisdom( win ); ) { 
        if( !keep_kgs ) { owi->kgs.clear(); } // no way to merge, so better not need later? or keep first? up to user.
        std::pair<by_op_set_p_op_wisdom_t::iterator,bool> ins_ret = all_wis.insert( owi );
        if( !ins_ret.second ) { (*ins_ret.first)->merge_wisdoms_from( *owi ); }
      }
    }
    virtual void main( nesi_init_arg_t * nia ) {
      p_ostream wout = ofs_open( wisdom_out_fn ); // open early to check for fn/fs errors
      for( vect_filename_t::const_iterator i = wisdom_in_fns.begin(); i != wisdom_in_fns.end(); ++i ) { proc_fn( *i ); }
      for( by_op_set_p_op_wisdom_t::const_iterator i = all_wis.begin(); i != all_wis.end(); ++i ) { write_op_wisdom( **i, *wout ); }
    }
  };

  
  struct filt_score_t {
    double tot_rt_secs;
    uint32_t tot_num;
    filt_score_t( void ) : tot_rt_secs(0), tot_num(0) {}
    void add_run( op_run_t const & r ) { ++tot_num; tot_rt_secs += r.rt_secs; }
  };

  
  typedef map< string, filt_score_t > map_str_filt_score_t;
  void run_system_cmd( string const &cmd, bool const verbose );

  struct per_op_ana_t {
    op_run_t const * min_r;
    p_op_tune_t min_tune;
    op_run_t const * ref_r;
    per_op_ana_t( void ) : min_r(0), ref_r(0) { }
  };
  typedef vector< per_op_ana_t > vect_per_op_ana_t; 

  struct wis_ana_t : virtual public nesi, public has_main_t // NESI(help="analyses wisdom file, output data in format for plotting",
           // bases=["has_main_t"], type_id="wis-ana" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="if true, include print results to stdout")
    filename_t wisdom_in_fn; //NESI(help="wisdom input file (to add to, may contain known-good results for checking)",req=1)
    p_filename_t csv_out_fn; //NESI(help="csv output filename")
    string csv_res_tag; //NESI(default="",help="suffix to add to csv results column names in header line")


    uint32_t s_img; //NESI(default="0",help="0 == all # of imgs; otherwise, only ops with the specified #")
    string s_plat; //NESI(default=".*",help="regex to select targ plat tag")

    map_str_uint32_t op_tunes;

    p_string ref_tune; //NESI(help="if specified, emit 'reference' tune times (and exclude this tune from selection for per-op-best)")

    by_op_set_p_op_wisdom_t all_wis;

    vect_per_op_ana_t per_op_anas;
    map_str_filt_score_t filt_scores;

    uint32_t show_aom; //NESI(default="1",help="if true, include aom in output")
    uint32_t show_pom; //NESI(default="1",help="if true, include pom in output")
    uint32_t show_ref; //NESI(default="1",help="if true, include pom in output")

    uint32_t run_wis_plot; //NESI(default="0",help="if true, run wis-plot.py (implicitly on csv output)")


    uint32_t get_tix( op_tune_t const & ot ) { return op_tunes.insert( make_pair( str(ot), op_tunes.size()+1 ) ).first->second; }

    // FIXME: cut-n-paste from cnn-prof ...
    uint64_t get_op_flops( p_op_base_t const & op ) {

      // assert( op->is( Convolution_coi ) ); // FIXME

      dims_t din;
      dims_t dout;
      uint64_t B;
      uint64_t M,N,K;
      //uint64_t forward_bytes;
      uint64_t forward_flops;

      dout = op->get_dims("out");
      din = op->get_dims("in");
      B = din.dsz( "img" );
      assert_st( B == dout.dsz("img" ) );
      // AI-related calculations
      dims_t const & filts = op->get_dims("filts");
      //dims_t const & biases = op->get_dims("biases");
      M = dout.dsz("img")*dout.dsz("x")*dout.dsz("y"); // note: all-imgs M
      K = filts.dsz("in_chan")*filts.dsz("x")*filts.dsz("y");
      N = filts.dsz("out_chan");
      //forward_bytes = (din.dims_prod() + dout.dims_prod() + filts.dims_prod() + biases.dims_prod()) * 4;
      forward_flops = M * N * K * 2;

      return forward_flops;
    }

    virtual void main( nesi_init_arg_t * nia ) {
      p_istream win = ifs_open( wisdom_in_fn );
      p_ostream csv_out = csv_out_fn ? ofs_open( *csv_out_fn ) : p_ostream();

      // print csv header
      if( csv_out ) { (*csv_out) << "OP FLOPS"; }
      if( csv_out && show_aom ) { (*csv_out) << " AOM" + csv_res_tag; }
      if( csv_out && show_pom ) { (*csv_out) << " POM" + csv_res_tag; }
      if( csv_out && show_ref ) { (*csv_out) << " REF" + csv_res_tag; }
      if( csv_out ) { (*csv_out) << "\n"; }

      regex r_plat( s_plat );
      for( p_op_wisdom_t owi; owi = read_next_wisdom( win ); ) { 
        owi->kgs.clear(); // no need for kgs
        if( s_img && (owi->op->get_dims("in").dsz("img") != s_img) ) { continue; } // filter by # imgs (permanent)
        std::pair<by_op_set_p_op_wisdom_t::iterator,bool> ins_ret = all_wis.insert( owi );
        assert_st( ins_ret.second ); // should be no dupe ops
        filter_runs( **ins_ret.first, r_plat );
      }

      filt_score_t best_fs;

      for( by_op_set_p_op_wisdom_t::const_iterator i = all_wis.begin(); i != all_wis.end(); ++i ) { 
        p_op_wisdom_t const & owi = *i;
        double min_time = std::numeric_limits<double>::max();
        per_op_anas.push_back( per_op_ana_t() );
        per_op_ana_t & poa = per_op_anas.back();
        for( vect_p_op_tune_wisdom_t::const_iterator otwi = owi->wisdoms.begin(); otwi != owi->wisdoms.end(); ++otwi ) {
          p_op_tune_t const & op_tune = (*otwi)->op_tune;
          for( map_str_op_run_t::const_iterator ri = (*otwi)->runs.begin(); ri != (*otwi)->runs.end(); ++ri ) {
            op_run_t const & r = ri->second;
            string const op_tune_str = str(op_tune);
            if( ref_tune && (op_tune_str == *ref_tune) ) { // if ref tune, store and exclude from min-tune set
              assert_st( !poa.ref_r ); // should be no dupe runs (i.e. only one-plat filts supported)
              poa.ref_r = &r;
            } else {
              filt_scores[op_tune_str].add_run( r );
              if( r.rt_secs < min_time ) { min_time = r.rt_secs; poa.min_r = &r; poa.min_tune = op_tune;}
            }
          }
        }        
        if( poa.min_r ) { best_fs.add_run( *poa.min_r ); }
      }
      assert_st( per_op_anas.size() == all_wis.size() ); // one-to-one mapping, in order

      // find best overall tune. FIXME: deal with differing #s of runs better ... normalize?
      
      filt_score_t const * min_filt_fs = 0;
      string min_filt_tune;
      for( map_str_filt_score_t::const_iterator i = filt_scores.begin(); i != filt_scores.end(); ++i ) {
        if( (!min_filt_fs) || 
            (i->second.tot_num > min_filt_fs->tot_num) || // first priority: max # of cases handled without error
            ( (i->second.tot_num == min_filt_fs->tot_num) && (i->second.tot_rt_secs < min_filt_fs->tot_rt_secs) ) ) {
          min_filt_fs = &i->second; min_filt_tune = i->first; 
        }
      }

      uint32_t poa_ix = 0;
      for( by_op_set_p_op_wisdom_t::const_iterator i = all_wis.begin(); i != all_wis.end(); ++i, ++poa_ix ) { 
        p_op_wisdom_t const & owi = *i;
        per_op_ana_t & poa = per_op_anas[poa_ix];
        if( verbose ) { printf( "owi->op=%s\n", str(owi->op).c_str() ); }
        if( csv_out ) { (*csv_out) << strprintf( "%s %s", str(owi->op).c_str(), str(get_op_flops(owi->op)).c_str() ); }

        if( show_aom ) {
          double v = NAN;
          for( vect_p_op_tune_wisdom_t::const_iterator otwi = owi->wisdoms.begin(); otwi != owi->wisdoms.end(); ++otwi ) {
            p_op_tune_t const & op_tune = (*otwi)->op_tune;
            for( map_str_op_run_t::const_iterator ri = (*otwi)->runs.begin(); ri != (*otwi)->runs.end(); ++ri ) {
              op_run_t const & r = ri->second;
              if( str(op_tune) == str(min_filt_tune) ) {
                v = r.rt_secs;
                if( verbose ) { 
                  printf( "  ALL-OP MIN: r.be_plat_tag=%s r.rt_secs=%s min_tune=%s\n", str(r.be_plat_tag).c_str(), str(r.rt_secs).c_str(), 
                          str(op_tune).c_str() );
                }
              }
            }
          }
          if( csv_out ) { (*csv_out) << strprintf( " %s", str(v).c_str() ); }
        }
        if( show_pom ) {
          double v = NAN;
          if( poa.min_r ) {
            op_run_t const & r = *poa.min_r;
            v = r.rt_secs;
            if( verbose ) {
              printf( "  PER-OP MIN: r.be_plat_tag=%s r.rt_secs=%s min_tune=%s\n", str(r.be_plat_tag).c_str(), str(r.rt_secs).c_str(), 
                      str(poa.min_tune).c_str() );
            }
          }
          if( csv_out ) { (*csv_out) << strprintf( " %s", str(v).c_str() ); }
        }
        if( show_ref ) {
          double v = NAN;
          if( poa.ref_r ) {
            op_run_t const & r = *poa.ref_r;
            v = r.rt_secs;
            if( verbose ) {
              printf( "  PER-OP REF: r.be_plat_tag=%s r.rt_secs=%s min_tune=%s\n", str(r.be_plat_tag).c_str(), str(r.rt_secs).c_str(), 
                      ref_tune->c_str() );
            }
          }
          if( csv_out ) { (*csv_out) << strprintf( " %s", str(v).c_str() ); }
        }
        if( csv_out ) { (*csv_out) << "\n"; }
      }
      if( csv_out ) { csv_out.reset(); } // flush/close
      if( run_wis_plot ) {
        string const title_str = strprintf( "%s (FWD-only, %s Images)", str(s_plat).c_str(), str(s_img).c_str() );
        string const cmd = "python ../../pysrc/wis-plot.py --title=\""+title_str+"\" out.csv";
        run_system_cmd( cmd, 1 );
      }
    }
  };

#include"gen/op-tuner.cc.nesi_gen.cc"

}
