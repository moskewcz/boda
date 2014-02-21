#ifndef BODA_BASE_H
#define BODA_BASE_H

#include<cassert>
#include<boost/shared_ptr.hpp>
#include<fstream>
#include<string>
#include<vector>
#include<map>
#include<set>
#include<exception>
#include<boost/integer_traits.hpp>

# define assert_st(expr)							\
  ((expr)								\
   ? __ASSERT_VOID_CAST (0)						\
   : __assert_fail ((std::string(" failed:\n") + __STRING(expr)+\
		     "\n "+stacktrace_str( get_backtrace(), 0 )).c_str(),	\
		    __FILE__, __LINE__, __ASSERT_FUNCTION))
namespace boost {
  namespace filesystem3 {
    struct path;
  }
  namespace iostreams {
    struct mapped_file;
    struct mapped_file_source;
    struct mapped_file_sink;
  }
}
namespace boda
{
  struct zi_uint32_t { uint32_t v; zi_uint32_t( void ) : v(0) { } };

  uint32_t const uint32_t_const_max = boost::integer_traits< uint32_t >::const_max;
  int32_t const int32_t_const_max = boost::integer_traits< int32_t >::const_max;
  int32_t const int32_t_const_min = boost::integer_traits< int32_t >::const_min;
  uint16_t const uint16_t_const_max = boost::integer_traits< uint16_t >::const_max;
  int16_t const int16_t_const_max = boost::integer_traits< int16_t >::const_max;
  int16_t const int16_t_const_min = boost::integer_traits< int16_t >::const_min;
  uint8_t const uint8_t_const_max = boost::integer_traits< uint8_t >::const_max;
  int8_t const int8_t_const_max = boost::integer_traits< int8_t >::const_max;
  int8_t const int8_t_const_min = boost::integer_traits< int8_t >::const_min;


  using boost::shared_ptr;
  using boost::dynamic_pointer_cast;
  using boost::iostreams::mapped_file;
  using boost::iostreams::mapped_file_source;
  using boost::iostreams::mapped_file_sink;
  using std::string;
  using std::vector;
  using std::map;
  using std::set;

  // NESI types
  struct filename_t { string in; string exp; };
  typedef vector< filename_t > vect_filename_t; 
  typedef shared_ptr< filename_t > p_filename_t; 
  typedef vector< p_filename_t > vect_p_filename_t;

  // typedef uint32_t nesi_bool // TODO

  typedef vector< string > vect_string;
  typedef shared_ptr< vect_string > p_vect_string;
  typedef set< string > set_string;

  typedef map< string, zi_uint32_t > map_str_ziu32_t;

  typedef shared_ptr< void > p_void;
  typedef shared_ptr< std::istream > p_istream;
  typedef shared_ptr< std::ifstream > p_ifstream;
  p_ifstream ifs_open( string const & fn );
  typedef shared_ptr< std::ostream > p_ostream;
  typedef shared_ptr< std::ofstream > p_ofstream;
  bool ifs_getline( std::string const &fn, p_ifstream in, std::string & line );
  p_ofstream ofs_open( string const & fn );
  p_vect_string readlines_fn( string const & fn );

  typedef shared_ptr< boost::iostreams::mapped_file > p_mapped_file;
  typedef shared_ptr< boost::iostreams::mapped_file_source > p_mapped_file_source;
  typedef shared_ptr< boost::iostreams::mapped_file_sink > p_mapped_file_sink;
  p_mapped_file_source map_file_ro( std::string const & fn );

  typedef vector< void * > vect_rp_void;
  typedef shared_ptr< vect_rp_void > p_vect_rp_void;
  p_vect_rp_void get_backtrace( void );
  string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames );
  bool ensure_is_dir( string const & fn, bool const create = 0 );
  void ensure_is_regular_file( string const & fn );

  typedef vector< uint8_t > vect_uint8_t;  
  typedef shared_ptr< uint8_t > p_uint8_t;  
  typedef vector< uint32_t > vect_uint32_t;
  typedef shared_ptr< uint32_t > p_uint32_t;  
  typedef vector< float > vect_float; 
  typedef shared_ptr< float > p_float;
  typedef vector< double > vect_double; 
  typedef shared_ptr< double > p_double; 

  void * posix_memalign_check( size_t const sz, uint32_t const a );
  // ma_p == make aligned shared_ptr. note: alloc'd with posix_memaling() and has custom deleter that invokes free()
  template< typename T > inline shared_ptr< T > ma_p_T( uint32_t const num, uint32_t const a ) {
    return shared_ptr< T >( (T *)posix_memalign_check( sizeof(T)*num, a), free );
  };
  template< typename T > inline shared_ptr< T > ma_p_T( uint32_t const num ) { 
    return ma_p_T<T>( num, std::max(sizeof(T),32UL) ); }
  // type-specific (syn. sugar)
  inline p_uint8_t ma_p_uint8_t( size_t const num, uint32_t const a ) { return ma_p_T<uint8_t>(num,a); }
  inline p_uint32_t ma_p_uint32_t( size_t const num, uint32_t const a ) { return ma_p_T<uint32_t>(num,a); }
  inline p_float ma_p_float( size_t const num, uint32_t const a ) { return ma_p_T<float>(num,a); }
  inline p_double ma_p_double( size_t const num, uint32_t const a ) { return ma_p_T<double>(num,a); }

  template< typename T > struct hunk_T {
    uint32_t sz; // in units of Ts
    shared_ptr< T > p;
    hunk_T( void ) : sz(0) { }
    hunk_T( shared_ptr< T > const & p_, uint32_t const sz_ ) : sz(sz_), p(p_) { }
    hunk_T( uint32_t const sz_ ) : sz(sz_), p( ma_p_T<T>( sz_ ) ) { } // hunk owns whole block
    T & operator []( uint32_t const & ix ) { assert_st( ix < sz ); return p.get()[ix]; }
    T const & operator []( uint32_t const & ix ) const { assert_st( ix < sz ); return p.get()[ix]; }
    uint64_t serial_sz( void ) const { return sizeof(uint32_t) + (sz * sizeof(T)); }
    bool operator == ( hunk_T<T> const & o ) const { 
      if( sz != o.sz ) { return 0; }
      for( uint32_t ix = 0; ix < sz; ++ix ) { if( (*this)[ix] != o[ix] ) { return 0; } }
      return 1;
    }
    uint32_t cnt_diff_elems( hunk_T<T> const & o ) const { 
      if( sz != o.sz ) { return uint32_t_const_max; }
      uint32_t ret = 0;
      for( uint32_t ix = 0; ix < sz; ++ix ) { if( (*this)[ix] != o[ix] ) { ++ret; } }
      return ret;
    }
  };

  template< typename RT, typename DT >
  void sum_squared_diffs( RT & ssd, RT & sd, DT const & o1, DT const & o2 ) {
    assert_st( o1.sz == o2.sz );
    for( uint32_t i = 0; i < o1.sz; ++i ) { 
      RT const d = RT(o1[i])-RT(o2[i]);
      sd += d; ssd += d*d;
    }
  }

  template< typename T >
  inline std::ostream & operator << ( std::ostream & out, hunk_T< T > const & o ) {
    out << "hunk of " << typeid(T).name() << " size=" << o.sz;
    for( uint32_t ix = 0; ix < o.sz; ++ix ) { out << " " << o[ix]; }
    out << std::endl;
    return out;
  }


  typedef hunk_T< uint32_t > hunk_uint32_t;
  typedef hunk_T< float_t > hunk_float_t;
  typedef hunk_T< double_t > hunk_double_t;

  uint32_t const boda_magic= *(uint32_t *)"BODA";

  template< typename T > inline void bwrite( std::ostream & out, T const & o ) { 
    out.write( (char *)&o, sizeof(o) ); assert_st( out.good() ); }
  template< typename T > inline void bread( std::istream & in, T & o ) { 
    in.read( (char *)&o, sizeof(o) ); assert_st( in.good() ); }

  template< typename T > inline void bwrite( std::ostream & out, shared_ptr< T > const & o ) { bwrite( out, *o ); }
  template< typename T > inline void bread( std::istream & in, shared_ptr< T > & o ) { 
    o.reset( new T );
    bread( in, *o );
  }

  
  template< typename T > inline void bwrite( std::ostream & out, vector< T > const & o ) { 
    uint32_t const sz = o.size();
    bwrite( out, sz );
    for( uint32_t i = 0; i < sz; ++i ) { bwrite( out, o[i] ); }
  }
  template< typename T > inline void bread( std::istream & in, vector< T > & o ) { 
    uint32_t sz = 0;
    bread( in, sz );
    o.resize( sz );
    for( uint32_t i = 0; i < sz; ++i ) { bread( in, o[i] ); }
  }

  inline void bwrite( std::ostream & out, string const & o ) {
    uint32_t const sz = o.size();
    bwrite( out, sz );
    out.write( (char *)&o[0], o.size()*sizeof(string::value_type) );
  }
  inline void bread( std::istream & in, string & o ) {
    uint32_t sz = 0;
    bread( in, sz );
    o.resize( sz );
    in.read( (char *)&o[0], o.size()*sizeof(string::value_type) );
    assert_st( in.good() );
  }

  inline void bwrite_id( std::ostream & out, string const & o ) { bwrite( out, string("id") ); bwrite( out, o ); }

  template< typename T > inline void bwrite( std::ostream & out, hunk_T< T > const & o ) { 
    bwrite( out, o.sz );
    out.write( (char *)o.p.get(), o.sz*sizeof(T) );
    assert_st( out.good() );
  }
  template< typename T > inline void bread( std::istream & in, hunk_T< T > & o ) { 
    bread( in, o.sz );
    o.p = ma_p_T<T>( o.sz );
    in.read( (char *)o.p.get(), o.sz*sizeof(T) );
    assert_st( in.good() );
  }


  // row-major dimensions holder
  // cm_ variants implicitly operate on the reversed dims list to provide column-major access
  // using less than the full # of dims uses a prefix (or suffix for the cm_ variants) of the dims
  struct dims_t {
    hunk_uint32_t dss; // dims and strides, strides first, in T units (i.e. not neccessarily bytes). can be set manually.
    bool operator == ( dims_t const & o ) const { return dss == o.dss; }
    bool fits_in( dims_t const & o ) const { 
      if( sz() != o.sz() ) { return 0; }
      for( uint32_t ix = 0; ix != sz(); ++ix ) { if( dims(ix) > o.dims(ix) ) { return 0; } }
      return 1;
    }

    void resize_and_zero( uint32_t const num_dims ) { 
      dss = hunk_uint32_t( num_dims*2 ); 
      for( uint32_t ix = 0; ix != num_dims; ++ix ) { dims(ix) = 0; strides(ix) = 0; }
    }
    uint32_t sz( void ) const { return dss.sz>>1; }
    uint32_t & dims( uint32_t const ix ) { return dss[ix+sz()]; }
    uint32_t & strides( uint32_t const ix ) { return dss[ix]; }
    uint32_t const & dims( uint32_t const ix ) const { return dss[ix+sz()]; }
    uint32_t const & strides( uint32_t const ix ) const { return dss[ix]; }

    uint32_t calc_strides( void ) { // assuming no padding, i.e. unit stride for fast dim and so on
      strides(sz()-1) = 1;
      for( uint32_t d = 1; d < sz(); ++d ) { strides(sz()-d-1) = strides(sz()-d)*dims(sz()-d); }
      return strides(0)*dims(0); // i.e. tot_elems, strides(-1), size of all data, used to bounds check
    }
    uint64_t serial_sz( void ) const { return dss.serial_sz(); }

    uint32_t cm_ix1( uint32_t const d0i ) const {
      return strides(sz()-1)*d0i; }
    uint32_t cm_ix2( uint32_t const d0i, uint32_t const d1i ) const { 
      return strides(sz()-1)*d0i + strides(sz()-2)*d1i; }
    uint32_t cm_ix3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      return strides(sz()-1)*d0i + strides(sz()-2)*d1i + strides(sz()-3)*d2i ; }
    uint32_t cm_ix4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      return strides(sz()-1)*d0i + strides(sz()-2)*d1i + strides(sz()-3)*d2i + strides(sz()-4)*d3i; }

    uint32_t ix( vect_uint32_t const & di ) const {
      assert_st( di.size() <= sz() );
      uint32_t ret = 0;
      for( uint32_t dix = 0; dix < di.size(); ++dix ) { ret += di[dix]*strides(dix); }
      return ret;
    }
    uint32_t ix1( uint32_t const d0i ) const {
      return d0i*strides(0); }
    uint32_t ix2( uint32_t const d0i, uint32_t const d1i ) const { 
      return d0i*strides(0) + d1i*strides(1); }
    uint32_t ix3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      return d0i*strides(0) + d1i*strides(1) + d2i*strides(2); }
    uint32_t ix4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      return d0i*strides(0) + d1i*strides(1) + d2i*strides(2) + d3i*strides(3); }
  };

  struct dims_iter_t {
    dims_t const & dims;
    vect_uint32_t di;
    dims_iter_t( dims_t const & dims_ ) : dims(dims_) {
      di.resize( dims.sz(), 0 );
    }
    bool next( void ) { // returns 0 if iteration is complete (leaving the iterator back at 0).
      uint32_t dix_end = di.size();
      while( dix_end ) { 
	if( ++di[dix_end-1] == dims.dims(dix_end-1) ) { di[dix_end-1] = 0; --dix_end; } 
	else { return 1; }
      }
      return 0;
    }
  };

  inline std::ostream & operator << ( std::ostream & out, dims_t const & o ) {
    out << "RM_DIMS=";
    for( uint32_t ix = 0; ix != o.sz(); ++ix ) { out << (ix?"*":"") << o.dims(ix); }
    out << " STRIDES=";
    for( uint32_t ix = 0; ix != o.sz(); ++ix ) { out << (ix?",":"") << o.strides(ix); }
    return out;
  }


  inline void bwrite( std::ostream & out, dims_t const & o ) { bwrite( out, o.dss ); }
  inline void bread( std::istream & in, dims_t & o ) { bread( in, o.dss ); }

  template< typename T > struct nda_T {
    hunk_T< T > elems; 
    dims_t dims;
    void set_dims( dims_t const & dims_ ) { 
      dims = dims_; 
      uint32_t strides_sz = 0;
      if( !dims.strides(0) ) { strides_sz = dims.calc_strides(); } 
      else { strides_sz = (dims.dims(0) * dims.strides(0)); }
      if( elems.p ) { assert( strides_sz == elems.sz ); }
      else { // if hunk is null, allocate it
	assert( !elems.sz ); // should not be set yet
	elems = hunk_T< T >( strides_sz );
      }
    }
    uint64_t serial_sz( void ) const { return dims.serial_sz() + elems.serial_sz(); }

    T & cm_at1( uint32_t const d0i ) { 
      uint32_t const ix = dims.cm_ix1( d0i ); return elems[ix]; }
    T & cm_at2( uint32_t const d0i, uint32_t const d1i ) { 
      uint32_t const ix = dims.cm_ix2( d0i,d1i ); return elems[ix]; }
    T & cm_at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      uint32_t const ix = dims.cm_ix3( d0i,d1i,d2i ); return elems[ix]; }
    T & cm_at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      uint32_t const ix = dims.cm_ix4( d0i,d1i,d2i,d3i ); return elems[ix]; }

    T & at( vect_uint32_t const & di ) { uint32_t const ix = dims.ix( di ); return elems[ix]; }
    T & at1( uint32_t const d0i ) { 
      uint32_t const ix = dims.ix1( d0i ); return elems[ix]; }
    T & at2( uint32_t const d0i, uint32_t const d1i ) { 
      uint32_t const ix = dims.ix2( d0i,d1i ); return elems[ix]; }
    T & at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      uint32_t const ix = dims.ix3( d0i,d1i,d2i ); return elems[ix]; }
    T & at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      uint32_t const ix = dims.ix4( d0i,d1i,d2i,d3i ); return elems[ix]; }

    T const & cm_at1( uint32_t const d0i ) const { 
      uint32_t const ix = dims.cm_ix1( d0i ); return elems[ix]; }
    T const & cm_at2( uint32_t const d0i, uint32_t const d1i ) const { 
      uint32_t const ix = dims.cm_ix2( d0i,d1i ); return elems[ix]; }
    T const& cm_at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      uint32_t const ix = dims.cm_ix3( d0i,d1i,d2i ); return elems[ix]; }
    T const& cm_at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      uint32_t const ix = dims.cm_ix4( d0i,d1i,d2i,d3i ); return elems[ix]; }

    T const& at( vect_uint32_t const & di ) const { uint32_t const ix = dims.ix( di ); return elems[ix]; }
    T const& at1( uint32_t const d0i ) const { 
      uint32_t const ix = dims.ix1( d0i ); return elems[ix]; }
    T const& at2( uint32_t const d0i, uint32_t const d1i ) const { 
      uint32_t const ix = dims.ix2( d0i,d1i ); return elems[ix]; }
    T const& at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      uint32_t const ix = dims.ix3( d0i,d1i,d2i ); return elems[ix]; }
    T const& at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      uint32_t const ix = dims.ix4( d0i,d1i,d2i,d3i ); return elems[ix]; }
  };

  template< typename T >
  inline std::ostream & operator << ( std::ostream & out, nda_T< T > const & o ) {
    out << "nda " << o.dims << std::endl;
    // print rows by iterating over dim with fastest/smallest stride (using column major indexing, of course)
    for( uint32_t rix = 0; o.dims.cm_ix2( 0, rix ) != o.elems.sz; ++rix ) {
      out << "row=" << rix << ":";
      for( uint32_t cix = 0; cix != o.dims.cm_ix2( 0, 1 ); ++cix ) {
	out << " " << o.cm_at2( cix, rix );
      }
      out << std::endl;
    }
    

    return out;
  }

  template< typename T > inline void bwrite( std::ostream & out, nda_T<T> const & o ) { 
    bwrite(out,o.dims); bwrite(out,o.elems); }
  template< typename T > inline void bread( std::istream & in, nda_T<T> & o ) { 
    bread(in,o.dims); bread(in,o.elems); }
  
  typedef nda_T< float > nda_float_t;

  typedef vector< nda_float_t > vect_nda_float_t; 
  typedef shared_ptr< nda_float_t > p_nda_float_t; 
  typedef vector< p_nda_float_t > vect_p_nda_float_t;

  typedef nda_T< double > nda_double_t;

  typedef vector< nda_double_t > vect_nda_double_t; 
  typedef shared_ptr< nda_double_t > p_nda_double_t; 
  typedef vector< p_nda_double_t > vect_p_nda_double_t;

  string ssds_str( p_nda_double_t o1, p_nda_double_t o2 ); // get a string with info from sum_squared_diffs(o1,o2)

  struct rt_exception : public std::exception
  {
    string err_msg;
    p_vect_rp_void bt;
    string what_str;
    rt_exception( string const & err_msg_, p_vect_rp_void bt_ );
    virtual ~rt_exception() throw () {}
    virtual char const * what( void ) const throw();
    string what_and_stacktrace( void ) const;
    int get_ret_code( void ) const;
  };
  // raise runtime error. captures stack and raises exception with
  // given string as error message. as a perf. compromise, while the
  // top frames of the stack are always captured, they are not
  // processed into a full trace at throw time.
  void rt_err( string const & err_msg ) __attribute__((noreturn));
  // like above, but assumes PyErr_Occured(). currently just calls
  // PyErr_Print() then calls rt_err(...)
  void rt_py_err( string const & err_msg ) __attribute__((noreturn));

  // for any class to have nesi support, it must inherit ": public
  // virtual nesi", and *declare* get_cinfo() (which will be generated
  // in the corresponding .nesi_gen.cc file for the class declaration)
  struct cinfo_t;
  struct nesi { virtual cinfo_t const * get_cinfo( void ) const = 0; };
  typedef shared_ptr< nesi > p_nesi;
}

#endif // BODA_BASE_H

