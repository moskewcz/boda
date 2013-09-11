#include"boda_tu_base.H"
#include"str_util.H"
#include<memory>
#include<execinfo.h>
#include<cxxabi.h>

namespace boda 
{

  using namespace std;


// opens a ifstream that will raise expections for all errors (not
// including eof). note: this function itself will raise if the open()
// fails.
  p_ifstream ifs_open( std::string const & fn )
  {
    p_ifstream ret( new ifstream );
    ret->open( fn.c_str() );
    if( ret->fail() )
    {
      printstr( stacktrace_str() );
      assert( 0 );
    }
    ret->exceptions( ifstream::failbit | ifstream::badbit );
    return ret;
  }

  uint32_t const max_frames = 256;


  string stacktrace_str( void )
  {
    string ret;

    // we could easily double bt until the trace fits, but for now
    // we'll assume something has gone wrong if the trace is >
    // max_frames and allow truncation.
    vect_rp_void bt;
    bt.resize( max_frames );
    int const bt_ret = backtrace( &bt[0], bt.size() );
    assert( bt_ret >= 0 );
    assert( uint32_t(bt_ret) <= bt.size() );
    ret += strprintf( "begin stack trace with num_frames=%s%s:\n", str(bt_ret).c_str(), 
		      (uint32_t(bt_ret) < bt.size())?"":" <WARNING: frames truncated, oldest lost>" );
    bt.resize( bt_ret );

    unique_ptr< char * > bt_syms( backtrace_symbols( &bt[0], bt.size() ) );

    for( uint32_t i = 0; i < bt.size(); ++i )
    {

/// BEGIN-PASTA
      // we assume i is a legal index for bt_syms. we can't check this.
      char *begin_name = 0, *begin_offset = 0, *end_offset = 0;
      // find parentheses and +address offset surrounding the mangled name:
      // ./module(function+0x15c) [0x8048a6d]
      for (char *p = bt_syms.get()[i]; *p; ++p)
      {
	if (*p == '(')
	  begin_name = p;
	else if (*p == '+')
	  begin_offset = p;
	else if (*p == ')' && begin_offset) {
	  end_offset = p;
	  break;
	}
      }

      if (begin_name && begin_offset && end_offset
	  && begin_name < begin_offset)
      {
	*begin_name++ = '\0';
	*begin_offset++ = '\0';
	*end_offset = '\0';

	// mangled name is now in [begin_name, begin_offset) and caller
	// offset in [begin_offset, end_offset). now apply
	// __cxa_demangle():
/// END-PASTA
	int dm_ret = 0;
	unique_ptr< char > dm_fn( abi::__cxa_demangle(begin_name, 0, 0, &dm_ret ) );

	if( dm_ret == 0 ) 
	{
	  ret += strprintf( "%s : %s+%s\n", bt_syms.get()[i], dm_fn.get(), begin_offset ); 
	}
	else 
	{
	  ret += strprintf( "%s : %s()+%s (DM_FAIL=%s)\n", bt_syms.get()[i], begin_name, begin_offset, str(dm_ret).c_str() );
	}
      }
      else
      {
	// couldn't parse the line? print the whole line.
	ret += strprintf("%s\n", bt_syms.get()[i]);
      }
    }
    return ret;
  }

}
