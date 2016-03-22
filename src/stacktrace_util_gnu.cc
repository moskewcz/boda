// Copyright (c) 2013-2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"stacktrace_util.H"
#include"str_util.H"
#include<execinfo.h>
#include<cxxabi.h>

namespace boda 
{
  uint32_t const max_frames = 64;

  p_vect_rp_void get_backtrace( void )
  {
    // we could easily double bt until the trace fits, but for now
    // we'll assume something has gone wrong if the trace is >
    // max_frames and allow truncation. this also (hopefully) limits
    // the cost of backtrace().
    p_vect_rp_void bt( new vect_rp_void );
    bt->resize( max_frames );
    int const bt_ret = backtrace( &bt->at(0), bt->size() );
    assert( bt_ret >= 0 );
    assert( uint32_t(bt_ret) <= bt->size() );
    bt->resize( bt_ret );
    return bt;
  }

  string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames )
  {
    string ret;
    assert( !bt->empty() );
    ret += strprintf( "----STACK TRACE (FRAMES=%s-%s)%s----\n", 
		      str(bt->size()).c_str(), str(strip_frames).c_str(), 
		      (bt->size() < max_frames)?"":" <WARNING: frames may be truncated (oldest lost)>" );
    std::unique_ptr< char * > bt_syms( backtrace_symbols( &bt->at(0), bt->size() ) );
    for( uint32_t i = strip_frames; i < bt->size(); ++i )
    {
      // it's a little unclear what should be assert()s here versus possible/ignorable/handlable cases
      // note: we can't use assert_st() here
      string sym( bt_syms.get()[i] );
      size_t const op_pos = sym.find('(');
      assert( op_pos != string::npos ); // can't find '(' to start symbol name
      assert( (op_pos+1) < sym.size() ); // '(' should not be last char. 
      sym[op_pos] = 0; // terminate object/file name part
      size_t const plus_pos = sym.find('+',op_pos);
      if( plus_pos != string::npos ) { // if there was an '+' (i.e. an offset part)
	assert( (plus_pos+1) < sym.size() ); // '+' should not be last char. 
	sym[plus_pos] = 0; // terminate sym_name part
      }
      size_t const cp_pos = sym.find(')',(plus_pos!=string::npos)?plus_pos:op_pos);
      assert( cp_pos != string::npos ); // can't find ')' to end symbol(+ maybe offset) part      
      assert( (cp_pos+1) < sym.size() ); // ')' should not be last char. 
      sym[cp_pos] = 0; // terminate sym_name (+ maybe offset) part

      char * sym_name = &sym[ op_pos + 1 ];
      int dm_ret = 0;
      std::unique_ptr< char > dm_fn( abi::__cxa_demangle(sym_name, 0, 0, &dm_ret ) );
      if( dm_ret == 0 ) { sym_name = dm_fn.get(); }
      ret += strprintf( "  %s(%s%s%s)%s\n", sym.c_str(), sym_name, 
			(plus_pos==string::npos)?"":"+",
			(plus_pos==string::npos)?"":(sym.c_str()+plus_pos+1),
			sym.c_str()+cp_pos+1 ); 
    }
    return ret;
  }
}
