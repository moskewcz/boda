// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"pyif.H"
#include"stacktrace_util.H"
#include<memory>
#include<boost/filesystem.hpp>
#include<boost/iostreams/device/mapped_file.hpp>
#include<sys/mman.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<cmath>
#include<float.h>
#include"ext/half.hpp"

void boda_assert_fail( char const * expr, char const * file, unsigned int line, char const * func ) throw() {
  fprintf(stderr,"boda: %s:%u: %s: Assertion failed: %s\n",file,line,func,expr);
  abort();
}

namespace boda 
{
  using std::get_deleter;
  using std::unique_ptr;
  using std::ifstream;
  using std::ofstream;
  using std::isnan;
  using boost::filesystem::path;
  using boost::filesystem::filesystem_error;

  static string py_boda_dir_static;
  static string py_boda_test_dir_static;
  string const & py_boda_dir( void ) { return py_boda_dir_static; }
  string const & py_boda_test_dir( void ) { return py_boda_test_dir_static; }


  std::string dims_t::ix_str( vect_uint32_t const & di, bool const inlcude_flat_ix ) const {
    assert_st( di.size() == sz() );
    string ret;
    for( uint32_t i = 0; i != sz(); ++i ) { 
      if( i ) { ret += ":"; }
      if( !names(i).empty() ) { ret += names(i) + "="; }
      ret += str( di[i] );
    }
    if( inlcude_flat_ix ) { ret += " (" + str(ix(di)) + ")"; }
    return ret;
  }
  
  std::string dims_t::pretty_str( void ) const { // omits strides, prints names near dims if they exist
    string ret;
    for( uint32_t i = 0; i != sz(); ++i ) { 
      if( i ) { ret += ":"; }
      if( !names(i).empty() ) { ret += names(i) + "="; }
      ret += str( dims(i) );
    }
    return ret;
  }

  // wraps output in ()'s, omits strides, prints names near dims if they exist, otherwise create anon name
  std::string dims_t::param_str( bool const & show_type ) const { 
    string ret;
    bool its_comma_time = 0;
    if( show_type && !tn.empty() ) { ret += "__tn__="+tn; its_comma_time = 1; }
    for( uint32_t i = 0; i != sz(); ++i ) { 
      if( its_comma_time ) { ret += ","; }
      its_comma_time = 1;
      string print_name = names(i);
      if( print_name.empty() ) { print_name = "dim_" + str(i); }
      ret += print_name + "=";
      ret += str( dims(i) );
    }
    return "(" + ret + ")";
  }

  void boda_dirs_init( void ) {
    path pse_dir_dir = read_symlink( path("/proc")/"self"/"exe" ).parent_path().parent_path();
    py_boda_dir_static = pse_dir_dir.string();
    py_boda_test_dir_static = (pse_dir_dir/"test").string();
  }

  // prints errno
  void neg_one_fail( int const & ret, char const * const func_name ) {
    if( ret == -1 ) { rt_err( strprintf( "%s() failed with errno=%s (%s)", func_name, str(errno).c_str(),
					 strerror(errno) ) ); }
  }

  void rt_err_errno( char const * const func_name ) { rt_err( strprintf( "%s failed with errno=%s (%s)", func_name, str(errno).c_str(),
									 strerror(errno) ) ); }


  string ssds_diff_t::basic_str( void ) const {
    return strprintf( "cnt=%s sum_squared_diffs=%s avg_abs_diff=%s max_abs_diff=%s "
		      "sum_diffs=%s avg_diff=%s", 
		      str( num_diff ).c_str(),
		      str( ssds ).c_str(), str( aad ).c_str(), str( mad ).c_str(), 
		      str( sds ).c_str(), str( ad ).c_str() ); 
  }
  std::ostream & operator <<(std::ostream & os, ssds_diff_t const & v) {
    os << v.basic_str();
    os << strprintf( " max_rel_diff=%s avg1=%s avg2=%s", 
		     str( v.mrd ).c_str(), str( v.avg1 ).c_str(), str( v.avg2 ).c_str() );
    return os;
  }

  // note: difference is o2 - o1, i.e. the delta/diff to get to o2 from o1. mad == max absolute diff
  template< typename T > struct ssds_diff_per_type_T : public ssds_diff_per_type_t {
    ssds_diff_t & v;
    T * o1;
    T * o2;
    ssds_diff_per_type_T( ssds_diff_t & v_ ) : v(v_) {
      o1 = static_cast<T *>(v.o1->rp_elems());
      o2 = static_cast<T *>(v.o2->rp_elems());
    }
    virtual void cnt_diff_elems( void ) { for( uint64_t i = 0; i < v.sz; ++i ) { if( o1[i] != o2[i] ) { ++v.num_diff; } } }
    virtual void sum_squared_diffs( void ) { 
      for( uint64_t i = 0; i < v.sz; ++i ) { 
        v.sum1 += double(o1[i]);
        v.sum2 += double(o2[i]);
        double const d = double(o2[i])-double(o1[i]); // difference
        v.sds += d; 
        v.ssds += d*d; 
        double const ad = (d<0)?-d:d; // absolute difference
        max_eq(v.mad,ad);
        double a1 = double(o1[i]);
        a1 = (a1<0)?-a1:a1;
        double a2 = double(o2[i]);
        a2 = (a2<0)?-a2:a2;
        double amax = std::max(a1,a2);
        // for values smaller than 1, clamp relative difference to be the value itself, in particular when compared
        // against 0 or very small values. that is, 1 vs 0 can have a relatve difference of 1, but 1e-1 vs 0 can only have
        // a rel diff of 1e-1 (not 1, as would be the case if we didn't clamp). if we set the relative error tolerance to,
        // say, 1e-5, then cases where both values are <= 1e-5 can only have a max rel diff of 1e-5. basically, we're
        // assuming that values with small absolute values (i.e. less than 1) are less important in terms of errors wrt
        // each other. so, i guess this is a hybrid between a relative and an absolute tolerance.
        amax = std::max(1.0,amax); 
        max_eq(v.mrd,ad/amax);
      }
    }
  };

  template< template<typename> class PT > struct make_per_type_t {
    ssds_diff_t & v;
    make_per_type_t( ssds_diff_t & v_ ) : v(v_) { }
    template< typename T > void operator()( void ) const { 
      v.pt = std::make_shared< PT<T> >( v );
    }
  };

  ssds_diff_t::ssds_diff_t( p_nda_t const & o1_, p_nda_t const & o2_ ) : o1(o1_), o2(o2_) {
    clear();
    sz = o1->elems_sz();
    assert_st( sz == o2->elems_sz() );
    assert_st( o1->dims.tn == o2->dims.tn );
    tn_dispatch( o1->dims.tn, make_per_type_t<ssds_diff_per_type_T>( *this ) );
    pt->sum_squared_diffs();
    pt->cnt_diff_elems();
    aad = sqrt(ssds / sz);
    ad = sds / sz;
    avg1 = sum1 / sz;
    avg2 = sum2 / sz;
  }

  bool ssds_diff_t::has_nan( void ) const { return isnan( ssds ) || isnan( sds ) || isnan( mad ); }

  // note: for some nda's, there are multiple param string that encode the same nda. in particular, redundant
  // specification of dims when it can be inferred, or tn when it is float and dims are specified, will yield such
  // cases. since we don't store information in the nda_t about which of the possible string forms it might have come
  // from (if it came from a string), we can't quite guarentee that round-triping an nda from param-str to nda to
  // param-str will give the same result on the first pass. but we always emit the 'cannonical' string form here (with
  // minimal information). luckily, there don't appear to be any cases where there's a non-obvious choice about what the
  // canonical form should be.
  struct nda_dump_t {
    std::ostream & out;
    nda_dump_t( std::ostream & out_ ) : out(out_) {}
    template< typename T > void op( nda_t const & nda ) const { 
      T const * const elems = static_cast<T const *>(nda.rp_elems());
      assert_st( elems );
      for( uint32_t i = 0; i != nda.elems_sz(); ++i ) { if( i ) { out << ":"; } out << elems[i]; }
    }
  };

  std::ostream & operator << ( std::ostream & out, nda_t const & o ) { 
    out << "(";
    // FIXME: put these 'default dims/tn for nda_t param str' conditions somewhere better?
    bool const show_dims = !( o.dims.empty() ); // omit (default) dims for scalars. note: always show dims for vectors.
    bool const show_tn = (!show_dims) || (o.dims.get_tn() != "float");
    if( show_tn ) { out << "tn=" << o.dims.get_tn(); }
    if( show_dims ) { 
      if( show_tn ) { out << ","; } 
      out << "dims=" << o.dims.param_str(0); // note: always omit showing tn (as __tn__ pseudo-dim) here
    }
    if( o.rp_elems() ) {
      out << ",v="; // note: will always emit at least one of tn and dims, so we always add a ',' here.
      nda_dispatch( o, nda_dump_t( out ) );
    }
    out << ")";
    return out; 
  }

  struct nda_c_const_str_t {
    std::ostream & out;
    nda_c_const_str_t( std::ostream & out_ ) : out(out_) {}
    template< typename T > void op( nda_t const & nda ) const { 
      T const * const elems = static_cast<T const *>(nda.rp_elems());
      assert_st( elems );
      for( uint32_t i = 0; i != nda.elems_sz(); ++i ) { if( i ) { out << ":"; } elem_c_const_str(elems[i]); }
    }
    template< typename T > void elem_c_const_str( T const & v ) const { out << v; }
    // i promise this is right! actually i have no idea; it looks complex enough to be right though, right?
    void elem_c_const_str( float const & v ) const { out << strprintf( "%#.9gf", v ); } 
  };


  string get_scalar_c_const_str( nda_t const & nda ) {
    assert_st( nda.elems_sz() == 1 );
    std::stringstream s;
    nda_dispatch( nda, nda_c_const_str_t(s) );
    return s.str();
  }

  struct nda_lt_elems_t {
    nda_t const & t;
    bool & ret;
    nda_lt_elems_t( nda_t const & t_, bool & ret_ ) : t(t_), ret(ret_) {}
    template< typename T > void op( nda_t const & o ) const { 
      assert_st( t.elems_sz() == o.elems_sz() );
      T const * const te = static_cast<T const *>(t.rp_elems());
      T const * const oe = static_cast<T const *>(o.rp_elems());
      for( uint32_t i = 0; i != t.elems_sz(); ++i ) { if( te[i] < oe[i] ) { ret=1; return; } }
    }
  };
  // note: more like shortlex then lexigographic order, since an nda with lesser dims (i.e. shorter) is less.
  bool nda_t::operator <( nda_t const & o ) const {
    // compare dims, then null-ness (null is less than non-null), then elements
    if( dims != o.dims ) { return dims < o.dims; }
    if( bool(d) != bool(o.d) ) { return bool(d) < bool(o.d); }
    if( !bool(d) ) { assert_st( !bool(o.d)); return 0; } // in fact, *this and o are equal here
    bool ret = 0;
    nda_dispatch( o, nda_lt_elems_t( *this, ret ) );
    return ret;
  }



  // questionably, we use (abuse?) the fact that we can mutate the
  // deleter to support mremap()ing the memory pointed to by the
  // deleter's (unique) shared_ptr.
  struct uint8_t_munmap_deleter { 
    size_t length;
    uint8_t_munmap_deleter( size_t const & length_ ) : length(length_) { assert(length); }
    void operator()( uint8_t * const & b ) const { if( !length ) { return; } if( munmap( b, length) == -1 ) { rt_err("munmap"); } } 
  };

  p_uint8_t make_mmap_shared_p_uint8_t( int const fd, size_t const length, off_t const offset ) {
    int flags = MAP_SHARED;
    if( fd == -1 ) { assert_st( !offset ); flags |= MAP_ANONYMOUS; }
    void * ret = mmap( 0, length, PROT_READ | PROT_WRITE, flags, fd, offset);
    if( MAP_FAILED == ret ) { rt_err_errno("mmap(...)"); }
    return p_uint8_t( (uint8_t *)ret, uint8_t_munmap_deleter( length ) ); 
  }

  void remap_mmap_shared_p_uint8_t( p_uint8_t &p, size_t const new_length ) {
    assert_st( p.unique() );
    uint8_t_munmap_deleter * d = get_deleter<uint8_t_munmap_deleter>(p);
    assert_st( d );
    assert( new_length > d->length );
    void * new_d = mremap( p.get(), d->length, new_length, MREMAP_MAYMOVE );
    if( MAP_FAILED == new_d ) { rt_err_errno("mmap(...)"); }
    d->length = 0; // make currently deleter into a no-op
    p.reset( (uint8_t *)new_d, uint8_t_munmap_deleter( new_length ) );
  }

  void * posix_memalign_check( size_t const sz, uint32_t const a ) {
    void * p = 0;
    int const ret = posix_memalign( &p, a, sz );
    if( ret ) { rt_err( strprintf( "posix_memalign( p, %s, %s ) failed, ret=%s", 
				   str(a).c_str(), str(sz).c_str(), str(ret).c_str() ) ); }
    return p;
  }

  bool ensure_is_dir( string const & fn, bool const create ) { 
    path const p(fn);
    try  { 
      bool const is_dir_ret = is_directory( p );
      if( (!create) && (!is_dir_ret) ) { 
	rt_err( strprintf("expected path '%s' to be a directory, but it is not.", p.c_str() ) ); 
      } 
      if( is_dir_ret ) { return 0; }
    } catch( filesystem_error const & e ) {
      rt_err( strprintf( "filesystem error while trying to check if '%s' is a directory: %s", 
			 p.c_str(), e.what() ) ); 
    }
    try  { // if we get here, p is not a directory, so try to create it
      bool const cd_ret = boost::filesystem::create_directory( p );
      assert_st( cd_ret == 1 ); // should not already be dir, so we should either create or raise an expection
      return 1;
    } catch( filesystem_error const & e ) {
      rt_err( strprintf( "error while trying to create '%s' directory: %s", 
			 p.c_str(), e.what() ) ); 
    }
  }
  void ensure_is_regular_file( string const & fn ) { 
    path const p( fn );
    try  { 
      bool const ret = is_regular_file( p ); 
      if( !ret ) { rt_err( strprintf("expected path '%s' to be a regular file, but it is not.", p.c_str()));}}
    catch( filesystem_error const & e ) {
      rt_err( strprintf( "filesystem error while trying to check if '%s' is a regular file: %s", 
			 p.c_str(), e.what() ) ); }
  }

  filename_t ensure_one_is_regular_file( filename_t const & fna, filename_t const & fnb ) {
    if( is_regular_file( path( fna.exp ) ) ) { return fna; }
    else if( is_regular_file( path( fnb.exp ) ) ) { return fnb; }
    else {
      rt_err( strprintf("neither file '%s' (expanded: '%s') nor alternate file '%s' (expanded: '%s') is a regular file",
			fna.in.c_str(), fna.exp.c_str(), fnb.in.c_str(), fnb.exp.c_str() ) ); 
    }
  }

  void set_fd_cloexec( int const fd, bool const val ) {
    int fd_flags = 0;
    neg_one_fail( fd_flags = fcntl( fd, F_GETFD ), "fcntl" );
    if( val ) { fd_flags |= FD_CLOEXEC; }
    else { fd_flags &= ~FD_CLOEXEC; }
    neg_one_fail( fcntl( fd, F_SETFD, fd_flags ), "fcntl" );
  }

  vect_rp_const_char get_vect_rp_const_char( vect_string const & v ) {
    vect_rp_const_char ret;
    for( vect_string::const_iterator i = v.begin(); i != v.end(); ++i ) { ret.push_back( (char *)i->c_str() ); }
    return ret;
  }
  vect_rp_char get_vect_rp_char( vect_string const & v ) {
    vect_rp_char ret;
    for( vect_string::const_iterator i = v.begin(); i != v.end(); ++i ) { ret.push_back( (char *)i->c_str() ); }
    return ret;
  }


  void fork_and_exec_self( vect_string const & args ) {
    vect_rp_char argp = get_vect_rp_char( args );
    argp.push_back( 0 );
    string const self_exe = py_boda_dir() + "/lib/boda"; // note: uses readlink on /proc/self/exe internally
    pid_t const ret = fork();
    if( ret == 0 ) {
      execve( self_exe.c_str(), &argp[0], environ );
      rt_err( strprintf( "execve() of '%s' failed. envp=environ, args=%s", self_exe.c_str(), str(args).c_str() ) );
    }
    // ret == child pid, not used
  }

  void fork_and_exec_cmd( vect_string const & args ) {
    vect_rp_char argp = get_vect_rp_char( args );
    argp.push_back( 0 );
    pid_t const ret = fork();
    if( ret == 0 ) {
      execvpe( argp[0], &argp[0], environ );
      printf( "*** boda: post-fork() execvpe() failed. envp=environ, args=%s ***"
              "\n*** boda: exiting child. parent will likely hang. ***\n", str(args).c_str() );
      exit(1);
    }
    // ret == child pid, not used
  }

  // opens a ifstream. note: this function itself will raise if the open() fails.
  p_ifstream ifs_open( filename_t const & fn )
  {
    ensure_is_regular_file( fn.exp );
    p_ifstream ret( new ifstream );
    ret->open( fn.exp.c_str() );
    if( ret->fail() ) { rt_err( strprintf( "can't open file '%s' for reading", fn.in.c_str() ) ); }
    assert( ret->good() );
    return ret;
  }
  p_ifstream ifs_open( std::string const & fn ) { return ifs_open( filename_t{fn,fn} ); }

  // clears line and reads one line from in. returns true if at EOF. 
  // note: calls rt_err() if a complete line cannot be read.
  bool ifs_getline( std::string const &fn, p_ifstream in, string & line )
  {
    line.clear();
    // the file should initially be good (including if we just
    // opened it).  note the eof is not set until trying to read
    // past the end. after each line is read, we check for eof, and
    // if we're not at eof, we check that the stream is still good
    // for more reading.
    assert_st( in->good() ); 
    getline(*in, line);
    if( in->eof() ) { 
      if( !line.empty() ) { rt_err( "reading "+fn+": incomplete (no newline) line at EOF:'" + line + "'" ); } 
      return 1;
    }
    else {
      if( !in->good() ) { rt_err( "reading "+fn+ " unknown failure" ); }
      return 0;
    }
  }


  p_vect_string readlines_fn( filename_t const & fn ) {
    p_ifstream in = ifs_open( fn );
    p_vect_string ret( new vect_string );
    string line;
    while( !ifs_getline( fn.in, in, line ) ) { ret->push_back( line ); }
    return ret;
  }
  p_vect_string readlines_fn( string const & fn ) { return readlines_fn( filename_t{fn,fn} ); }

  // opens a ofstream. note: this function itself will raise if the open() fails.
  p_ofstream ofs_open( filename_t const & fn )
  {
    p_ofstream ret( new ofstream );
    ret->open( fn.exp.c_str() );
    if( ret->fail() ) { rt_err( strprintf( "can't open file '%s' for writing", fn.in.c_str() ) ); }
    assert( ret->good() );
    return ret;
  }
  p_ofstream ofs_open( std::string const & fn ) { return ofs_open( filename_t{fn,fn} ); }

  p_mapped_file_source map_file_ro( filename_t const & fn ) {
    //ensure_is_regular_file( fn ); // too strong? a good idea?
    p_mapped_file_source ret;
    try { ret.reset( new mapped_file_source( fn.exp ) ); }
    catch( std::exception & err ) { 
      // note: fn.c_str(),err.what() is not too useful? it does give 'permission denied' sometimes, but other times is it just 'std::exception'.
      rt_err( strprintf("failed to open/map file '%s' (expanded: '%s') for reading",fn.in.c_str(), fn.exp.c_str() ) ); 
    }
    assert_st( ret->is_open() ); // possible?
    return ret;
  }
  p_mapped_file_source map_file_ro( std::string const & fn ) { return map_file_ro( filename_t{fn,fn} ); }

  p_string read_whole_fn( filename_t const & fn ) {
    p_mapped_file_source mfile = map_file_ro( fn );
    uint8_t const * const fn_data = (uint8_t const *)mfile->data();
    return p_string( new string( fn_data, fn_data+mfile->size() ) );
  }
  p_string read_whole_fn( std::string const & fn ) { return read_whole_fn( filename_t{fn,fn} ); }
  void write_whole_fn( filename_t const & fn, std::string const & data ) {
    p_ofstream out = ofs_open( fn );
    (*out) << data;
  }
  void write_whole_fn( std::string const & fn, std::string const & data ) { return write_whole_fn( filename_t{fn,fn}, data ); }

  rt_exception::rt_exception( std::string const & err_msg_, p_vect_rp_void bt_ ) : err_msg(err_msg_), bt(bt_) {}
  char const * rt_exception::what( void ) const throw() { return err_msg.c_str(); }
  string rt_exception::what_and_stacktrace( void ) const { return err_msg + "\n" + stacktrace_str( bt, 2 ); }
  int rt_exception::get_ret_code( void ) const { return 1; }
  void rt_err( std::string const & err_msg ) { throw rt_exception( "error: " + err_msg, get_backtrace() ); }

}
