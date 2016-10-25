// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"op-tuner.H"
#include"str_util.H"
#include"nesi.H"
#include"lexp.H"
#include<sstream>

namespace boda 
{


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

}
