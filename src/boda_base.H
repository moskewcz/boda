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
  };

  typedef hunk_T< uint32_t > hunk_uint32_t;

  // row-major dimensions holder
  // cm_ variants implicitly operate on the reversed dims list to provide column-major access
  // using less than the full # of dims uses a prefix (or suffix for the cm_ variants) of the dims
  struct dims_t {
    hunk_uint32_t dims; // in arbitrary units (i.e. not neccessarily bytes). can be set manually.
    hunk_uint32_t strides; // in arbitrary units (i.e. not neccessarily bytes). can be set manually.
    void resize_and_zero( uint32_t const num_dims ) { dims = hunk_uint32_t( num_dims ); strides = hunk_uint32_t(); }
    uint32_t calc_strides( void ) { // assuming no padding, i.e. unit stride for fast dim and so on
      assert( dims.sz );
      strides = hunk_uint32_t( dims.sz );
      strides[strides.sz-1] = 1;
      for( uint32_t d = 1; d < dims.sz; ++d ) { strides[dims.sz-d-1] = strides[dims.sz-d]*dims[dims.sz-d]; }
      return strides[0]*dims[0]; // i.e. tot_elems, strides[-1], size of all data, used to bounds check
    }
    uint64_t serial_sz( void ) const { return dims.serial_sz() + strides.serial_sz(); }

    uint32_t cm_ix1( uint32_t const d0i ) {
      assert( strides.sz == dims.sz ); assert( strides.sz > 0 ); 
      return strides[strides.sz-1]*d0i; }
    uint32_t cm_ix2( uint32_t const d0i, uint32_t const d1i ) { 
      assert( strides.sz == dims.sz ); assert( strides.sz > 1 ); 
      return strides[strides.sz-1]*d0i + strides[strides.sz-2]*d1i; }
    uint32_t cm_ix3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      assert( strides.sz == dims.sz ); assert( strides.sz > 2 ); 
      return strides[strides.sz-1]*d0i + strides[strides.sz-2]*d1i + strides[strides.sz-3]*d2i ; }
    uint32_t cm_ix4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      assert( strides.sz == dims.sz ); assert( strides.sz > 3 ); 
      return strides[strides.sz-1]*d0i + strides[strides.sz-2]*d1i + strides[strides.sz-3]*d2i + strides[strides.sz-4]*d3i; }

    uint32_t ix1( uint32_t const d0i ) {
      assert( strides.sz == dims.sz ); assert( strides.sz > 0 ); 
      return d0i*strides[0]; }
    uint32_t ix2( uint32_t const d0i, uint32_t const d1i ) { 
      assert( strides.sz == dims.sz ); assert( strides.sz > 1 ); 
      return d0i*strides[0] + d1i*strides[1]; }
    uint32_t ix3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      assert( strides.sz == dims.sz ); assert( strides.sz > 2 ); 
      return d0i*strides[0] + d1i*strides[1] + d2i*strides[2]; }
    uint32_t ix4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      assert( strides.sz == dims.sz ); assert( strides.sz > 3 ); 
      return d0i*strides[0] + d1i*strides[1] + d2i*strides[2] + d3i*strides[3]; }
  };
  //struct nda_double_t { vect_uint32_t dims; p_double d; };

  template< typename T > struct nda_T {
    hunk_T< T > elems; 
    dims_t dims;
    //nda_T( shared_ptr< T > const & p_, uint32_t const sz_, uint32_t const off_ ) : elems(p_,sz_) { }
    //nda_T( hunk_T< T > const &hunk ) : elems(hunk) { }
    void set_dims( dims_t const & dims_ ) { 
      dims = dims_; 
      uint32_t strides_sz = 0;
      if( !dims.strides.p ) { strides_sz = dims.calc_strides(); } 
      else { strides_sz = (dims.dims[0] * dims.strides[0]); }

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

    T & at1( uint32_t const d0i ) { 
      uint32_t const ix = dims.ix1( d0i ); return elems[ix]; }
    T & at2( uint32_t const d0i, uint32_t const d1i ) { 
      uint32_t const ix = dims.ix2( d0i,d1i ); return elems[ix]; }
    T & at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      uint32_t const ix = dims.ix3( d0i,d1i,d2i ); return elems[ix]; }
    T & at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      uint32_t const ix = dims.ix4( d0i,d1i,d2i,d3i ); return elems[ix]; }
  };

  template< typename T >
  inline std::ostream & operator << ( std::ostream & out, nda_T< T > const & nda ) { 
    for( uint32_t ix = nda.off; ix < nda.sz; ++ix ) {
      if( ix != nda.off ) { out << " "; }
      out << nda.elems[ix];
    }
    return out;
  }
  
  typedef nda_T< float > nda_float_t;

  typedef vector< nda_float_t > vect_nda_float_t; 
  typedef shared_ptr< nda_float_t > p_nda_float_t; 
  typedef vector< p_nda_float_t > vect_p_nda_float_t;

  typedef nda_T< double > nda_double_t;

  typedef vector< nda_double_t > vect_nda_double_t; 
  typedef shared_ptr< nda_double_t > p_nda_double_t; 
  typedef vector< p_nda_double_t > vect_p_nda_double_t;

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

