// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef BODA_BASE_H
#define BODA_BASE_H

#include<cassert>
#include<memory>
#include<fstream>
#include<string>
#include<vector>
#include<map>
#include<set>
#include<exception>
#include<poll.h>
#include<boost/integer_traits.hpp>
#include<math.h>
#include<type_traits>

void boda_assert_fail( char const * expr, char const * file, unsigned int line, char const * func )
  throw() __attribute__ ((__noreturn__));

#define assert_st(expr) ((expr) ? static_cast<void>(0) : boda_assert_fail( (std::string("\n") + #expr + \
		     "\n "+stacktrace_str( get_backtrace(), 0 )).c_str(), __FILE__, __LINE__, __PRETTY_FUNCTION__ ) )

namespace half_float { class half; }
namespace boost {
  namespace iostreams {
    class mapped_file;
    class mapped_file_source;
    class mapped_file_sink;
  }
}
namespace boda
{
  struct zi_uint32_t { uint32_t v; zi_uint32_t( void ) : v(0) { } };
  struct zi_uint64_t { uint64_t v; zi_uint64_t( void ) : v(0) { } };
  struct zi_bool { bool v; zi_bool( void ) : v(0) { } zi_bool( bool const & v_ ) : v(v_) { } };

  uint64_t const uint64_t_const_max = boost::integer_traits< uint64_t >::const_max;
  uint32_t const uint32_t_const_max = boost::integer_traits< uint32_t >::const_max;
  int32_t const int32_t_const_max = boost::integer_traits< int32_t >::const_max;
  int32_t const int32_t_const_min = boost::integer_traits< int32_t >::const_min;
  uint16_t const uint16_t_const_max = boost::integer_traits< uint16_t >::const_max;
  int16_t const int16_t_const_max = boost::integer_traits< int16_t >::const_max;
  int16_t const int16_t_const_min = boost::integer_traits< int16_t >::const_min;
  uint8_t const uint8_t_const_max = boost::integer_traits< uint8_t >::const_max;
  int8_t const int8_t_const_max = boost::integer_traits< int8_t >::const_max;
  int8_t const int8_t_const_min = boost::integer_traits< int8_t >::const_min;
  float const float_const_max = std::numeric_limits<float>::max();
  float const float_const_min = std::numeric_limits<float>::lowest(); // note: ::min() is the lowest pos value ... sigh/sorry!

  // note: returns 0 for v==0
  inline uint64_t floor_log2_u64( uint64_t v ) { uint64_t ret = 0; while( v >>= 1 ) { ++ret; } return ret; }

  template< typename T > inline T clamp( T const & v, T const & min_v, T const & max_v  ) { return std::min( std::max( v, min_v ), max_v ); }
  template< typename T > inline void clamp_eq( T & v, T const & min_v, T const & max_v  ) { v = std::min( std::max( v, min_v ), max_v ); }
  template< typename T > inline void min_eq( T & v, T const & o ) { v = std::min( v, o ); }
  template< typename T > inline void max_eq( T & v, T const & o ) { v = std::max( v, o ); }
  inline uint32_t u32_ceil_div( uint32_t const & val, uint32_t const & d ) { return (val + d - 1) / d; }
  inline uint32_t u32_ceil_sqrt( uint32_t const v ) { return uint32_t( ceil( sqrt( double( v ) ) ) ); }

  inline uint32_t u32_ceil_align( uint32_t const & val, uint32_t const & align ) {
    return u32_ceil_div( val, align ) * align; }
  inline int32_t i32_floor_div( int32_t const & val, int32_t const & d ) { 
    return (val / d) - ( ( d > 0 ) ? ((val % d) < 0) : ((val % d) > 0)  ); 
  }
  inline int32_t i32_ceil_div( int32_t const & val, int32_t const & d ) { 
    return (val / d) + ( ( d < 0 ) ? ((val % d) < 0) : ((val % d) > 0)  ); 
  }
  inline int32_t u32_to_i32( uint32_t const & v ) { int32_t const ret( v ); assert( ret >= 0 ); return ret; }

  using std::shared_ptr;
  using std::make_shared;
  template< typename T > struct null_deleter { void operator()( T * const ) const {}; };
  using std::dynamic_pointer_cast;
  using boost::iostreams::mapped_file;
  using boost::iostreams::mapped_file_source;
  using boost::iostreams::mapped_file_sink;
  using std::string;
  using std::vector;
  using std::map;
  using std::set;
  using half_float::half;

  typedef vector< void * > vect_rp_void;
  typedef shared_ptr< vect_rp_void > p_vect_rp_void;
  typedef vector< char * > vect_rp_char;
  typedef vector< char const * > vect_rp_const_char;
  string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames );
  p_vect_rp_void get_backtrace( void );

  template< typename T > inline std::string str(T const & i); // predeclare inline function template from std_util.H

  // raise runtime error. captures stack and raises exception with
  // given string as error message. as a perf. compromise, while the
  // top frames of the stack are always captured, they are not
  // processed into a full trace at throw time.
  void rt_err( string const & err_msg ) __attribute__((noreturn));
  // like above, but assumes PyErr_Occured(). currently just calls
  // PyErr_Print() then calls rt_err(...)
  void rt_py_err( string const & err_msg ) __attribute__((noreturn));
  void rt_err_errno( char const * const func_name ); // like rt_err, but prints errno

  // similar to rt_err, but throws a derived class for some non-fatal (catchable) cases
  void unsup_err( string const & err_msg ) __attribute__((noreturn));

  template< typename T > void must_insert( T & kvm, typename T::key_type const & k ) {
    bool const did_ins = kvm.insert( k ).second;
    assert_st( did_ins );
  }		      
  template< typename T > void must_insert( T & kvm, typename T::key_type const & k, typename T::mapped_type const & v ) {
    bool const did_ins = kvm.insert( std::make_pair( k, v ) ).second;
    assert_st( did_ins );
  }		      
  template< typename T > void must_replace( T & kvm, typename T::key_type const & k, typename T::mapped_type const & v ) {
    typename T::iterator i = kvm.find( k );
    if( i == kvm.end() ) { rt_err( "missing key-to-replace:" + str(k) ); }
    i->second = v;
  }		      
  template< typename T > void must_erase( T & kvm, typename T::key_type const & k ) {
    uint32_t const num_del = kvm.erase( k );
    assert_st( num_del == 1 );
  }		      
  template< typename T > bool has( T const & kvm, typename T::key_type const & k ) { return kvm.find( k ) != kvm.end(); }
  template< typename T > bool vect_has( T const & vect, typename T::value_type const & v ) { 
    return std::find( vect.begin(), vect.end(), v ) != vect.end(); }
  template< typename T > typename T::mapped_type const & must_find( T const & kvm, typename T::key_type const & k ) {
    typename T::const_iterator i = kvm.find( k );
    if( i == kvm.end() ) { rt_err( "missing key:" + str(k) ); }
    return i->second;
  }
  template< typename T > typename T::mapped_type & must_find( T & kvm, typename T::key_type const & k ) {
    typename T::iterator i = kvm.find( k );
    if( i == kvm.end() ) { rt_err( "missing key:" + str(k) ); }
    return i->second;
  }

  template< typename T > vector< typename T::key_type > get_keys( T & kvm ) {
    vector< typename T::key_type > keys;
    for( typename T::const_iterator i = kvm.begin(); i != kvm.end(); ++i ) { keys.push_back( i->first ); }
    return keys;
  }

  template< typename T > typename T::mapped_type const & get( T const & kvm, typename T::key_type const & k, typename T::mapped_type const & dv ) {
    typename T::const_iterator i = kvm.find( k );
    if( i == kvm.end() ) { return dv; }
    return i->second;
  }

  template< typename T > typename T::mapped_type get_and_rem( T & kvm, typename T::key_type const & k, typename T::mapped_type const & dv ) {
    typename T::const_iterator i = kvm.find( k );
    if( i == kvm.end() ) { return dv; }
    typename T::mapped_type ret = i->second;
    kvm.erase( i );
    return ret;
  }

  // NESI types
  struct filename_t { string in; string exp; };
  typedef vector< filename_t > vect_filename_t; 
  typedef shared_ptr< filename_t > p_filename_t; 
  typedef vector< p_filename_t > vect_p_filename_t;

  // typedef uint32_t nesi_bool // TODO

  typedef vector< string > vect_string;
  typedef vector< vect_string > vect_vect_string;
  typedef shared_ptr< string > p_string;
  typedef shared_ptr< vect_string > p_vect_string;
  vect_rp_char get_vect_rp_char( vect_string const & v );
  vect_rp_const_char get_vect_rp_const_char( vect_string const & v );
  typedef std::pair< string, string > pair_str_str;
  typedef vector< pair_str_str > vect_pair_str_str;

  typedef set< string > set_string;
  typedef set< uint32_t > set_uint32_t;
  typedef set< uint64_t > set_uint64_t;
  typedef shared_ptr< set_uint64_t > p_set_uint64_t;

  typedef map< string, uint32_t > map_str_uint32_t;
  typedef map< string, zi_uint32_t > map_str_ziu32_t;
  typedef map< string, double > map_str_double;
  typedef map< string, string > map_str_str;
  typedef map< string, vect_string > map_str_vect_string;
  typedef map< uint64_t, uint64_t > map_uint64_t_uint64_t;
  struct dims_t;
  typedef map< string, dims_t > map_str_dims_t;

  template< typename T > struct check_T { typedef void void_; typedef int int_; };

  template< typename MAP, typename check_T<typename MAP::mapped_type>::int_ = 0 > 
  std::ostream & operator <<(std::ostream & os, MAP const & v) {
    os << "(";
    for( typename MAP::const_iterator i = v.begin(); i != v.end(); ++i ) {
      if( i != v.begin() ) { os << ","; }
      os << i->first << "=" << i->second;
    }
    os << ")";
    return os;
  }

  typedef shared_ptr< void > p_void;
  typedef shared_ptr< std::istream > p_istream;
  typedef shared_ptr< std::ifstream > p_ifstream;
  p_ifstream ifs_open( string const & fn );
  p_ifstream ifs_open( filename_t const & fn );
  typedef shared_ptr< std::ostream > p_ostream;
  typedef vector< p_ostream > vect_p_ostream;

  bool ifs_getline( std::string const &fn, p_istream const & in, std::string & line );
  p_ostream ofs_open( filename_t const & fn );
  p_ostream ofs_open( string const & fn );
  p_string read_whole_fn( filename_t const & fn );
  p_string read_whole_fn( string const & fn );
  void write_whole_fn( filename_t const & fn, std::string const & data );
  void write_whole_fn( string const & fn, std::string const & data );

  p_vect_string readlines_fn( filename_t const & fn );
  p_vect_string readlines_fn( string const & fn );

  void neg_one_fail( int const & ret, char const * const func_name );
  void fork_and_exec_self( vect_string const & args );
  void fork_and_exec_cmd( vect_string const & args );
  void set_fd_cloexec( int const fd, bool const val );

  typedef shared_ptr< boost::iostreams::mapped_file > p_mapped_file;
  typedef shared_ptr< boost::iostreams::mapped_file_source > p_mapped_file_source;
  typedef shared_ptr< boost::iostreams::mapped_file_sink > p_mapped_file_sink;
  p_mapped_file_source map_file_ro( filename_t const & fn );
  p_mapped_file_source map_file_ro( std::string const & fn );

  bool ensure_is_dir( string const & fn, bool const create = 0 );
  void ensure_is_regular_file( string const & fn );
  filename_t ensure_one_is_regular_file( filename_t const & fna, filename_t const & fnb );

  typedef vector< uint8_t > vect_uint8_t;  
  typedef shared_ptr< uint8_t > p_uint8_t;  
  typedef vector< p_uint8_t > vect_p_uint8_t;  
  typedef vector< uint32_t > vect_uint32_t;
  typedef shared_ptr< uint32_t > p_uint32_t;  
  typedef vector< int32_t > vect_int32_t;
  typedef vector< uint64_t > vect_uint64_t;  
  typedef shared_ptr< vect_uint64_t > p_vect_uint64_t;
  typedef vector< float > vect_float; 
  typedef shared_ptr< float > p_float;
  typedef vector< double > vect_double; 
  typedef shared_ptr< double > p_double; 

  p_uint8_t make_mmap_shared_p_uint8_t( int const fd, size_t const length, off_t const offset );
  void remap_mmap_shared_p_uint8_t( p_uint8_t &p, size_t const new_length );
  void * posix_memalign_check( size_t const sz, uint32_t const a );
  uint32_t const boda_default_align = 4096; // yep.
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

  __attribute__((used)) uint32_t const boda_magic= *(uint32_t *)"BODA";

  struct p_uint8_with_sz_t : public p_uint8_t {
    uint64_t sz;
    p_uint8_with_sz_t( uint8_t * & p, uint64_t const & sz_ ) : p_uint8_t( p ), sz(sz_) {}
    template< typename DELETER > p_uint8_with_sz_t( uint8_t * & p, uint64_t const & sz_, DELETER const & deleter ) : p_uint8_t( p, deleter ), sz(sz_) {}
  };

  // the b{read,write}_bytes() functions are the basic low-level read/write that must be implimented
  // for any stream class we wish to use for IO. the two basic ones here are for istream/ostream and
  // for ASIO classes that ASIO read/write will work on. the total # of actual instantiations is
  // probably pretty low: just istream and ostream for the IOS case, and a few socket/file classes
  // for ASIO. so we could explicitly instantiate them somewhere maybe.
  template< typename IOSWritable, typename check_T<typename IOSWritable::pos_type>::int_ = 0 > void
  bwrite_bytes( IOSWritable & out, char const * const & d, size_t const & sz ) {
    out.write( d, sz ); assert_st( out.good() ); }
  template< typename IOSReadable, typename check_T<typename IOSReadable::pos_type>::int_ = 0 > void
  bread_bytes( IOSReadable & in, char * const & d, size_t const & sz ) { in.read( d, sz ); assert_st( in.good() ); }
#if 1
  // FIXME: can't define here since asio isn't availible ... and we seem to need to declare them
  // before the below usages or the overloads won't be found? so we declare here, and define in the
  // files that will instantiate these ... hmm.
  template< typename AsioWritable, typename check_T<typename AsioWritable::lowest_layer_type>::int_ = 0 > void 
  bwrite_bytes( AsioWritable & out, char const * const & d, size_t const & sz ); //write( out, buffer( d, sz ) ); }
  template< typename AsioReadable, typename check_T<typename AsioReadable::lowest_layer_type>::int_ = 0 > void 
  bread_bytes( AsioReadable & in, char * const & d, size_t const & sz ); // read( in, buffer( d, sz ) ); }

  // similarly, for mapped file reader (and anthing else with has_bread_bytes, we need the relevant pre-declaration
  template< typename HasBreadBytes, typename check_T<typename HasBreadBytes::has_bread_bytes>::int_ = 0 > void 
  bread_bytes( HasBreadBytes & in, char * const & d, size_t const & sz );

#endif

  template< typename T > struct breadwrite_raw_T { };
  template<> struct breadwrite_raw_T<bool>{ typedef int int_; };
  template< typename T > struct T_pt_t; template<> struct breadwrite_raw_T<T_pt_t<uint32_t> >{ typedef int int_; };
  template<> struct breadwrite_raw_T<uint8_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<uint16_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<uint32_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<uint64_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<int8_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<int16_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<int32_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<int64_t>{ typedef int int_; };
  template<> struct breadwrite_raw_T<half>{ typedef int int_; };
  template<> struct breadwrite_raw_T<float>{ typedef int int_; };
  template<> struct breadwrite_raw_T<double>{ typedef int int_; };
  
  template< typename STREAM, typename T, typename breadwrite_raw_T<T>::int_ = 0 > 
  inline void bwrite( STREAM & out, T const & o ) { bwrite_bytes( out, (char *)&o, sizeof(o) ); }
  template< typename STREAM, typename T, typename breadwrite_raw_T<T>::int_ = 0 > 
  inline void bread( STREAM & in, T & o ) { bread_bytes( in, (char *)&o, sizeof(o) ); }

  template< typename STREAM > inline void bwrite( STREAM & out, string const & o ) {
    uint32_t const sz = o.size();
    bwrite( out, sz );
    if( sz ) { bwrite_bytes( out, (char *)&o[0], o.size()*sizeof(string::value_type) ); }
  }
  template< typename STREAM > inline void bread( STREAM & in, string & o ) {
    uint32_t sz = 0;
    bread( in, sz );
    o.resize( sz );
    if( sz ) { bread_bytes( in, (char *)&o[0], o.size()*sizeof(string::value_type) ); }
  }
  
  // this bit 'o magic allows us to use enable_if<> based on the presence/absence of a typedef it's pretty easy to just
  // use the typedef if we want to *enable* an overload, but not so much to disable one.  since we're (unwisely?) using
  // SFINAE for enabling/disabling function overloads for dispatching the proper shared_ptr<T> bread/bwrite global
  // functions, we need to ensure exactly one (ambiguous) overload is enabled per class. fun times!
  template<typename T, typename FOO = void > struct has_static_bread_and_virt_bwrite_t : std::false_type {};
  template<typename T> struct has_static_bread_and_virt_bwrite_t< T, typename check_T<typename T::has_static_bread_and_virt_bwrite>::void_ > : std::true_type {};

  template< typename STREAM, typename T > inline 
  typename std::enable_if< !has_static_bread_and_virt_bwrite_t<T>::value >::type
  bwrite( STREAM & out, shared_ptr< T > const & o ) { 
    uint8_t non_null = bool(o);
    bwrite( out, non_null );
    if( non_null ) { bwrite( out, *o ); }
  }
  template< typename STREAM, typename T > inline 
  typename std::enable_if< !has_static_bread_and_virt_bwrite_t<T>::value >::type
  bread( STREAM & in, shared_ptr< T > & o ) { 
    uint8_t non_null = 0;
    bread( in, non_null );
    if( non_null ) {
      o.reset( new T );
      bread( in, *o );
    } else { o.reset(); }
  }

  // special-case pointer IO for base classes where we read/write a derived type. note: for p_nda_digest_t is currently
  // the only such type, but if/when we add others, we should use this as a model.
  // note: T::has_static_bread_and_virt_bwrite is either void or non-existant; just using it directly as the return type
  // is simpler and equivalent to the below usage of enable_if<>+helper, but inconsistent with ! case
  template< typename STREAM, typename T > inline 
  typename std::enable_if< has_static_bread_and_virt_bwrite_t<T>::value >::type
  bwrite( STREAM & out, shared_ptr< T > const & o ) { 
    uint8_t non_null = bool(o);
    bwrite( out, non_null ); 
    if( non_null ) { o->bwrite( out ); } // note: 'out' better be an ostream ...
  }
  template< typename STREAM, typename T > inline 
  typename std::enable_if< has_static_bread_and_virt_bwrite_t<T>::value >::type
  bread( STREAM & in, shared_ptr< T > & o ) { 
    uint8_t non_null = 0;
    bread( in, non_null );
    if( non_null ) { o = T::bread( in ); } // note: 'in' better be an istream ...
    else { o.reset(); }
  }

  template< typename STREAM, typename T > inline void bwrite( STREAM & out, vector< T > const & o ) { 
    uint32_t const sz = o.size();
    bwrite( out, sz );
    for( uint32_t i = 0; i < sz; ++i ) { bwrite( out, o[i] ); }
  }
  template< typename STREAM, typename T > inline void bread( STREAM & in, vector< T > & o ) { 
    uint32_t sz = 0;
    bread( in, sz );
    o.resize( sz );
    for( uint32_t i = 0; i < sz; ++i ) { bread( in, o[i] ); }
  }

  template< typename STREAM, typename T, 
            typename check_T<typename T::mapped_type>::int_ = 0,
            typename check_T<typename T::key_type>::int_ = 0 >
  inline void bwrite( STREAM & out, T const & o ) { 
    uint32_t const sz = o.size();
    bwrite( out, sz );
    for( typename T::const_iterator i = o.begin(); i != o.end(); ++i ) {
      bwrite( out, i->first );
      bwrite( out, i->second );
    }
  }
  template< typename STREAM, typename T,
            typename check_T<typename T::mapped_type>::int_ = 0,
            typename check_T<typename T::key_type>::int_ = 0 >
  inline void bread( STREAM & in, T & o ) { 
    uint32_t sz = 0;
    bread( in, sz );
    for( uint32_t i = 0; i < sz; ++i ) { 
      typename T::key_type k;
      bread( in, k ); 
      typename T::mapped_type v;
      bread( in, v );
      must_insert( o, k, v );
    }
  }

  template< typename STREAM >
  inline void bwrite_id( STREAM & out, string const & o ) { bwrite( out, string("id") ); bwrite( out, o ); }

  // (note: experimental/unfinished) as opposed to dims_t/nda_float_t, these struct are intended for metaprogramming
  // use, and thus compactness and speed are less of a concern, and convenience/functionality more of a concern.
  struct dim_t {
    uint32_t sz; // in elements; sz=0 is unspecified/wildcard
    uint32_t stride; // in elements; stride=0 is uncomupted (/ treat as unpadded)
    // uint32_t pad_sz; // in bytes? elements?
    string name; // if empty, unspecified/wildcard
    dim_t( void ) : sz(0), stride(0) { }
    explicit dim_t( string const & name_ ) : sz(0), stride(0), name(name_) { }
    explicit dim_t( uint32_t const & sz_ ) : sz(sz_), stride(0) { }
    dim_t( string const & name_, uint32_t const & sz_ ) : sz(sz_), stride(0), name(name_) { }
    bool operator == ( dim_t const & o ) const { return (sz==o.sz) && (stride==o.stride) && (name==o.name); }
    bool operator != ( dim_t const & o ) const { return !(*this==o); }
    bool operator < ( dim_t const & o ) const { 
      if( sz != o.sz ) { return sz < o.sz; }
      if( stride != o.stride ) { return stride < o.stride; }
      if( name != o.name ) { return name < o.name; }
      return 0;
    }
    // like ==, but treats zero/empty sz/stride/names as wild in the passed template/'o' dim_t (NOT in the *this dim_t)
    bool matches_template( dim_t const & o ) const { // note: o is the template, not *this
      if( o.sz && (sz != o.sz) ) { return 0; }
      if( o.stride && (stride != o.stride) ) { return 0; }
      if( (!o.name.empty()) && (name != o.name) ) { return 0; }
      return 1;
    }
    bool has_sz_and_stride_and_name( void ) const { return sz && stride && (!name.empty()); }
    bool has_name( void ) const { return (!name.empty()); }
  };
  typedef vector< dim_t > vect_dim_t; 

  // oh, what type to choose for these fields? size_t(signed, but 'official'), uint8_t (right-sized), uint32_t
  // (fast/compact-ish and big enough)? sigh. i'm still not on the size_t bandwagon, and i'm worried about dealing with
  // upcasts for size computations where 32 bits isn't enough, and surely 64 bits is enough forever (and fast/native
  // enough), so uint64_t it is.
  // FIXME: remove some of the string comparisons in favor of ndat pointer comparisons? tempting, but probably not needed.
  struct ndat_info_t {
    string tn;
    uint64_t sz;
    bool is_float;
    bool is_signed;
  };
  typedef ndat_info_t const * rp_ndat_info_t;
  // dims-only ndas that can never actually point to any specific type of data. FIXME: rename "none" to "void"?
  ndat_info_t const none_ndat{"none",0,0,0}; 
  ndat_info_t const half_ndat{"half",2,1,1};
  ndat_info_t const float_ndat{"float",4,1,1};
  ndat_info_t const double_ndat{"double",8,1,1};
  ndat_info_t const int32_t_ndat{"int32_t",4,0,1};
  ndat_info_t const uint32_t_ndat{"uint32_t",4,0,0};
  ndat_info_t const uint16_t_ndat{"uint16_t",2,0,0};
  ndat_info_t const uint8_t_ndat{"uint8_t",1,0,0};
  template< typename T > inline ndat_info_t const & get_ndat( void );
  template<> inline ndat_info_t const & get_ndat<void>( void ) { return none_ndat; }
  template<> inline ndat_info_t const & get_ndat<half>( void ) { return half_ndat; }
  template<> inline ndat_info_t const & get_ndat<float>( void ) { return float_ndat; }
  template<> inline ndat_info_t const & get_ndat<double>( void ) { return double_ndat; }
  template<> inline ndat_info_t const & get_ndat<uint32_t>( void ) { return uint32_t_ndat; }
  template<> inline ndat_info_t const & get_ndat<uint16_t>( void ) { return uint16_t_ndat; }
  template<> inline ndat_info_t const & get_ndat<int32_t>( void ) { return int32_t_ndat; }
  template<> inline ndat_info_t const & get_ndat<uint8_t>( void ) { return uint8_t_ndat; }
  typedef map< string, rp_ndat_info_t > ndat_infos_t;
  ndat_infos_t const ndat_infos{
    {none_ndat.tn,&none_ndat},
    {half_ndat.tn,&half_ndat},
    {float_ndat.tn,&float_ndat},
    {double_ndat.tn,&double_ndat},
    {int32_t_ndat.tn,&int32_t_ndat},
    {uint32_t_ndat.tn,&uint32_t_ndat},
    {uint16_t_ndat.tn,&uint16_t_ndat},
    {uint8_t_ndat.tn,&uint8_t_ndat},
  };

  // row-major dimensions holder; consists of a vector of dim_t's and a type string.
  // cm_ variants implicitly operate on the reversed dims list to provide column-major access
  // using less than the full # of dims uses a prefix (or suffix for the cm_ variants) of the dims
  struct dims_t : public vect_dim_t {
    // ah, the irony of code like this in a code generation framework ... sigh.
    void add_dims( string const & name_, uint32_t const & sz_  ) { push_back( dim_t( name_, sz_ ) ); }
    void add_dims( string const & name1_, uint32_t const & sz1_, string const & name2_, uint32_t const & sz2_ ) { 
      add_dims( name1_, sz1_ ); add_dims( name2_, sz2_ ); }
    void add_dims( string const & name1_, uint32_t const & sz1_, string const & name2_, uint32_t const & sz2_,
		   string const & name3_, uint32_t const & sz3_ ) {
      add_dims( name1_, sz1_ ); add_dims( name2_, sz2_ ); add_dims( name3_, sz3_ ); }
    void add_dims( string const & name1_, uint32_t const & sz1_, string const & name2_, uint32_t const & sz2_,
		   string const & name3_, uint32_t const & sz3_, string const & name4_, uint32_t const & sz4_ ) { 
      add_dims( name1_, sz1_ ); add_dims( name2_, sz2_ ); add_dims( name3_, sz3_ ); add_dims( name4_, sz4_ ); }

    string tn;  // empty for wildcard; otherwise "float", "uint32_t", "int8_t", etc ...
    ndat_info_t const & tinfo( void ) const { return *must_find( ndat_infos, tn ); }
    uint64_t tsz( void ) const { return tinfo().sz; }
        

    //hunk_uint32_t dss; // dims and strides, strides first, in T units (i.e. not neccessarily bytes). can be set manually.
    dims_t clone( void ) const { return *this; }

    // note: == and < compare: dims *and* strides *and* names
    bool operator == ( dims_t const & o ) const { return (tn==o.tn)&&(((vect_dim_t const &)(*this)) == ((vect_dim_t const &)(o)));}
    bool operator != ( dims_t const & o ) const { return !(*this==o); }
    bool operator < ( dims_t const & o ) const { 
      return (tn==o.tn)?(((vect_dim_t const &)(*this)) < ((vect_dim_t const &)(o))):(tn<o.tn); }
    // like ==, but treats zero/empty sz/stride/names as wild in the passed template/'o' dims_t (NOT in the *this dims_t)
    bool matches_template( dims_t const & o ) const { // note: o is the template, not *this
      if( o.has_tn() && (tn != o.tn) ) { return 0; }
      if( sz() != o.sz() ) { return 0; }
      for( uint32_t i = 0; i != sz(); ++i ) { if( !at(i).matches_template( o.at(i) ) ) { return 0; } }
      return 1;
    }
    bool has_tn( void ) const { return tn.size(); }
    string get_tn( void ) const { return tn; } // maybe replace with tinfo().tn later, if/when we replace tn -> tinfo
    bool has_sz_and_stride_and_name( void ) const { 
      for( uint32_t i = 0; i != sz(); ++i ) { if( !at(i).has_sz_and_stride_and_name() ) { return 0; } }
      return 1;
    }
    bool has_name( void ) const { 
      for( uint32_t i = 0; i != sz(); ++i ) { if( !at(i).has_name() ) { return 0; } }
      return 1;
    }

    bool fits_in( dims_t const & o ) const {
      assert_st( o.tn == tn ); // FIXME: too strong?
      if( sz() != o.sz() ) { return 0; }
      for( uint32_t ix = 0; ix != sz(); ++ix ) { if( dims(ix) > o.dims(ix) ) { return 0; } }
      return 1;
    }
    dims_t( void ) { }
    explicit dims_t( uint32_t const num_dims ) { resize_and_zero( num_dims ); }
    dims_t( vect_uint32_t const & dims_, string const & tn_ ) { init( dims_, 0, tn_ ); }
    dims_t( vect_uint32_t const & dims_, vect_string const & names_, string const & tn_ ) { init( dims_, &names_, tn_ ); }
    void init( vect_uint32_t const & dims_, vect_string const * const names_, string const & tn_ ) {
      tn = tn_;
      assert_st( !tn.empty() );
      resize_and_zero( dims_.size() ); 
      for( uint32_t i = 0; i != dims_.size(); ++i ) { dims(i) = dims_[i]; }
      if( names_ ) {
	assert_st( names_->size() == size() );
	for( uint32_t i = 0; i != size(); ++i ) { names(i) = names_->at(i); }
      }
      calc_strides();
    }
    dim_t const * get_dim_by_name( string const & name ) const {
      for( uint32_t i = 0; i != sz(); ++i ) { if( names(i) == name ) { return &at(i); } }
      return 0; // not found
    }
    dim_t const & must_get_dim_by_name( string const & name ) const {
      for( uint32_t i = 0; i != sz(); ++i ) { if( names(i) == name ) { return at(i); } }
      rt_err( "dim not found:" + name );
    }
    dim_t & must_get_dim_by_name( string const & name ) {
      for( uint32_t i = 0; i != sz(); ++i ) { if( names(i) == name ) { return at(i); } }
      rt_err( "dim not found:" + name );
    }
    uint32_t dsz( string const & name ) const { return must_get_dim_by_name( name ).sz; } 
    uint32_t dstride( string const & name ) const { return must_get_dim_by_name( name ).stride; }
    void resize_and_zero( uint32_t const num_dims ) { resize(num_dims); }

    dims_t operator - ( dims_t const & o ) const { 
      assert_st( o.fits_in( *this ) );
      dims_t ret( *this );
      for( uint32_t ix = 0; ix != sz(); ++ix ) { ret.dims(ix) -= o.dims(ix); }
      ret.calc_strides();
      return ret;
    }
    uint32_t sz( void ) const { return size(); }
    uint64_t bytes_sz( void ) const { return tsz()*strides_sz; }

    uint32_t & dims( uint32_t const ix ) { return at(ix).sz; }
    string & names( uint32_t const ix ) { return at(ix).name; }
    uint32_t & strides( uint32_t const ix ) { return at(ix).stride; }

    uint32_t const & dims( uint32_t const ix ) const { return at(ix).sz; }
    string const & names( uint32_t const ix ) const { return at(ix).name; }
    uint32_t const & strides( uint32_t const ix ) const { return at(ix).stride; }

    std::string pretty_str( void ) const; // omits strides, prints names near dims if they exist
    std::string param_str( bool const & show_type ) const; // in NESI/param format, see comment at def.

    void clear_dims( void ) { for( uint32_t i = 0; i != sz(); ++i ) { dims(i) = 0; } }
    void clear_strides( void ) { for( uint32_t i = 0; i != sz(); ++i ) { strides(i) = 0; } }
    bool has_any_strides( void ) const { for( uint32_t i = 0; i != sz(); ++i ) { if( strides(i) ) { return 1; } } return 0; }
    bool has_any_names( void ) const { for( uint32_t i = 0; i != sz(); ++i ) { if( !names(i).empty() ) { return 1; } } return 0; }

    void make_dims_vect( vect_uint32_t & dv ) const { 
      dv.resize(sz()); for( uint32_t ix = 0; ix != sz(); ++ix ) { dv[ix] = dims(ix); } }
    // aka 'number of elements' / size / count. 
    // note: doesn't use strides(0)*dims(0) (which wouldn't be generally right).
    uint32_t dims_prod( void ) const { uint32_t ret = 1; for( uint32_t ix = 0; ix != sz(); ++ix ) { ret *= dims(ix); } return ret; }
    //uint32_t nda_sz( void ) { uint32_t ret = 1; for( vect_dim_t::const_iterator i = begin(); i != end(); ++i ) { ret *= i->sz; } return ret; }

    // return true if the strides are different from the no-padding-strides as would be calculated by calc_strides()
    // FIXME: untested
    bool has_padding( void ) const {
      uint32_t stride;
      for( uint32_t d = 0; d < sz(); ++d ) { 
	stride = d ? (stride * dims(sz()-d)) : 1;
	if( stride != strides(sz()-d-1) ) { return 1; }
      }
      return 0;
    }

    // note: calc_strides() can/must be called if any dims are modified after dims_t creation, since all the strides
    // become invalid/stale in that case.
    uint64_t strides_sz; // aka strides(-1), stride of all data; in no-padding case == dims_prod()
    zi_bool strides_valid;
    // FIXME: cache dims_prod() here too?
    void calc_strides( bool const is_template = 0 ) { // assuming no padding, i.e. unit stride for fast dim and so on
      if( !is_template ) {
        assert_st( !tn.empty() ); // FIXME: good place for this?
      }
      // FIXME: do something different in is_template case here?
      strides_sz = 1;
      for( uint32_t d = 0; d < sz(); ++d ) { 
        strides(sz()- d - 1) = strides_sz; strides_sz *= dims(sz() - d - 1);
      }
      // note: strides_sz is now tot_elems, strides(-1), size of all data, used to bounds check (same as dims_prod())
      strides_valid.v = 1;
    }
    bool valid( void ) const {
      if( tn.empty() ) { return 0; }
      if( !strides_valid.v ) { return 0; } // strides must be set (and non-zero)
      return 1;
    }
    uint32_t cm_ix1( uint32_t const d0i ) const {
      return strides(sz()-1)*d0i; }
    uint32_t cm_ix2( uint32_t const d0i, uint32_t const d1i ) const { 
      return strides(sz()-1)*d0i + strides(sz()-2)*d1i; }
    uint32_t cm_ix3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      return strides(sz()-1)*d0i + strides(sz()-2)*d1i + strides(sz()-3)*d2i ; }
    uint32_t cm_ix4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      return strides(sz()-1)*d0i + strides(sz()-2)*d1i + strides(sz()-3)*d2i + strides(sz()-4)*d3i; }

    std::string ix_str( vect_uint32_t const & di, bool const inlcude_flat_ix ) const;

    uint32_t ix( vect_uint32_t const & di ) const {
      assert_st( di.size() <= sz() );
      uint32_t ret = 0;
      for( uint32_t dix = 0; dix < di.size(); ++dix ) { ret += di[dix]*strides(dix); }
      return ret;
    }
    uint32_t ix_sum( vect_uint32_t const & di, vect_uint32_t const & di2 ) const {
      assert_st( di.size() <= sz() );
      assert_st( di2.size() == di.size() );
      uint32_t ret = 0;
      for( uint32_t dix = 0; dix < di.size(); ++dix ) { ret += (di[dix]+di2[dix])*strides(dix); }
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

  typedef vector< dims_t > vect_dims_t; 

  // note: iterates over last/fast dim innermost.
  struct dims_iter_t {
    vect_uint32_t b;
    vect_uint32_t e;
    vect_uint32_t di;
    dims_iter_t( dims_t const & dims_ ) {
      b.resize( dims_.sz(), 0 );
      dims_.make_dims_vect( e );
      di = b;
    }
    dims_iter_t( dims_t const & b_, dims_t const & e_ ) {
      b_.make_dims_vect( b );
      e_.make_dims_vect( e );
      di = b;
    }
    bool next( void ) { // returns 0 if iteration is complete (leaving the iterator back at b).
      uint32_t dix_end = di.size();
      while( dix_end ) { 
	if( ++di[dix_end-1] == e[dix_end-1] ) { di[dix_end-1] = b[dix_end-1]; --dix_end; } 
	else { return 1; }
      }
      return 0;
    }
  };


  inline dims_t make_scalar_dims_t( string const & tn ) { return dims_t{ {}, tn }; }
  inline dims_t make_vector_dims_t( string const & tn, uint64_t const sz ) { return dims_t{ {(uint32_t)sz}, {"v"}, tn }; } // FIXME_32
  // valid: scalar with "none" type. not commonly used. FIXME: remove, replace with ... no-data nda with null dims maybe?
  inline dims_t make_null_dims_t( void ) { return make_scalar_dims_t("none"); } 

  inline std::ostream & operator << ( std::ostream & out, dims_t const & o ) { out << o.param_str(1); return out; }

  template< typename STREAM > inline void bwrite( STREAM & out, dim_t const & o ) { 
    bwrite( out, o.sz ); bwrite( out, o.stride ); bwrite( out, o.name ); 
  }
  template< typename STREAM > inline void bread( STREAM & in, dim_t & o ) {
    bread( in, o.sz ); bread( in, o.stride ); bread( in, o.name );  
  }

  template< typename STREAM > inline void bwrite( STREAM & out, dims_t const & o ) { 
    bwrite( out, (vect_dim_t const &)o ); 
    bwrite( out, o.tn );
    bwrite( out, o.strides_sz );
    bwrite( out, o.strides_valid.v );
  }
  template< typename STREAM > inline void bread( STREAM & in, dims_t & o ) { 
    bread( in, (vect_dim_t &)o ); 
    bread( in, o.tn );
    bread( in, o.strides_sz );
    bread( in, o.strides_valid.v );
  }

  struct nda_t {
    nda_t( void ) { }
    nda_t( dims_t const & dims_ ) { nda_set_dims( dims_ ); alloc(); }
    nda_t( dims_t const & dims_, p_uint8_t const &d_ ) { nda_set_dims( dims_ ); d = d_; } // create from existing data held by p_uint8_t
    // below ctor is for no-ownership-transfer references to 'remote' NDAs where the data is behine an opaque void *
    nda_t( dims_t const & dims_, void * const & elems_ ) { 
      d.reset( (uint8_t *)elems_, null_deleter<uint8_t>() ); 
      nda_set_dims( dims_ ); 
      // note: no alloc, even if d was explicitlt set to null (which is the dims-only/no-data case)
    }
    void const * rp_elems( void ) const { return (void const *)d.get(); }
    void * rp_elems( void ) { return (void *)d.get(); }
    p_uint8_t const & get_internal_data( void ) { return d; } // be careful with this ...
  protected:
    p_uint8_t d;
  public:
    uint64_t const elems_sz( void ) const { return dims.strides_sz; }
    dims_t dims;
    bool operator ==( nda_t const & o ) const = delete; // not hard to impl if needed
    bool operator !=( nda_t const & o ) const = delete;
    bool operator <( nda_t const & o ) const;
  protected:
    void alloc( void ) {
      assert_st( !d );
      assert_st( dims.strides_sz );
      d = ma_p_T<uint8_t>(elems_sz()*dims.tsz(),32UL);
    }
    void nda_set_dims( dims_t const & dims_ ) { 
      dims = dims_;
      assert_st( dims.valid() );
    }
  public:
    template< typename STREAM > void bwrite_( STREAM & out ) const {
      bwrite(out,dims); 
      uint8_t non_null = bool(d);
      bwrite( out, non_null );
      if( non_null ) { bwrite_bytes( out, (char *)d.get(), dims.bytes_sz() ); }
    }
    template< typename STREAM > void bread_( STREAM & in ) {
      bread(in,dims); 
      uint8_t non_null = 0;
      bread( in, non_null );
      if( non_null ) {
        alloc();
        bread_bytes( in, (char *)d.get(), dims.bytes_sz() );
      } else {
        assert_st( !d );
      }
    }
    // similarly to bread/bwrite, the nesi init code needs to take a default constructed nda, fill in the dims, and alloc.
    void nesi_init_private_alloc( void ) { alloc(); } // wrapper to make it clear this should not be called willy-nilly
  };
  std::ostream & operator << ( std::ostream & out, nda_t const & o );
  template< typename STREAM > inline void bwrite( STREAM & out, nda_t const & o ) { o.bwrite_( out ); }
  template< typename STREAM > inline void bread( STREAM & in, nda_t & o ) { o.bread_( in ); }

  template< typename T > void tn_dispatch( string const & tn, T const & func ) {
    if( 0 ) { }
    // note: in particular, we *don't* handle the void/none type here, so trying to nda_dispatch on them will fail. but,
    // since such ndas can never have elements, we shouldn't ever try to do that, since nda_dispatch is (ony) used to
    // operate/iterate on the elements of an nda.
    else if( tn == half_ndat.tn ) { func.template operator()<half>(); } // C++!
    else if( tn == float_ndat.tn ) { func.template operator()<float>(); }
    else if( tn == double_ndat.tn ) { func.template operator()<double>(); }
    else if( tn == int32_t_ndat.tn ) { func.template operator()<int32_t>(); }
    else if( tn == uint32_t_ndat.tn ) { func.template operator()<uint32_t>(); }
    else if( tn == uint16_t_ndat.tn) { func.template operator()<uint16_t>(); }
    else if( tn == uint8_t_ndat.tn) { func.template operator()<uint8_t>(); }
    else { rt_err( "unhandled type in tn_dispatch; tn=" + tn ); } 
  }

  template< typename NDA_T, typename FUNC > struct nda_dispatch_helper_t {
    NDA_T & nda;
    FUNC const & func;
    nda_dispatch_helper_t( NDA_T & nda_, FUNC const & func_ ) : nda(nda_), func(func_) {}
    template< typename T > void operator()( void ) const { func.template op<T>( nda ); }
  };

  template< typename NDA_T, typename T > // below magic return type is void and makes NDA_T be const or non-const nda_t
  typename std::enable_if< std::is_same<nda_t, typename std::remove_const<NDA_T>::type >::value >::type // C++! yeah! sigh.
  nda_dispatch( NDA_T & nda, T const & func ) {
    nda_dispatch_helper_t<NDA_T,T> ndh( nda, func );
    tn_dispatch( nda.dims.tn, ndh );
  }
  
  typedef vector< nda_t > vect_nda_t; 
  typedef shared_ptr< nda_t > p_nda_t; 
  typedef vector< p_nda_t > vect_p_nda_t;
  typedef map< string, p_nda_t > map_str_p_nda_t;
  typedef shared_ptr< map_str_p_nda_t > p_map_str_p_nda_t;
  typedef vector< p_map_str_p_nda_t > vect_p_map_str_p_nda_t;

  template< typename T > T nda_at( p_nda_t const & nda, dims_iter_t const & di ) {
    assert_st( get_ndat<T>().tn == nda->dims.tn );
    uint64_t const ix = nda->dims.ix( di.di );
    assert_st( ix < nda->elems_sz() );
    return (static_cast<T *>(nda->rp_elems()))[ix];
  }

  template< typename T > inline T scalar_nda_extract( nda_t const & nda ) {
    string const & want_tn = get_ndat<T>().tn;
    if( want_tn != nda.dims.tn ) { rt_err( "can't extract scalar of type '"+want_tn+"' from nda of type '" + nda.dims.tn + "'." ); }
    if( nda.elems_sz() != 1 ) { rt_err( "can't extract scalar from nda with "+str(nda.elems_sz())+" elems." ); }
    if( !nda.rp_elems() ) { rt_err( "can't extract scalar from null/no-data nda" ); }
    return *(static_cast< T const * >(nda.rp_elems()));
  }
  template< typename T > inline T SNE( nda_t const & nda ) { return scalar_nda_extract<T>( nda ); } // short name


  template< typename T > inline p_nda_t make_scalar_nda( T const & v ) { 
    p_nda_t ret = make_shared<nda_t>( make_scalar_dims_t( get_ndat<T>().tn ) );
    assert_st( ret->elems_sz() == 1);
    *(static_cast< T * >(ret->rp_elems())) = v;
    return ret;
  }
  template< typename T > inline p_nda_t make_vector_nda( vector< T > const & vv ) { 
    p_nda_t ret = make_shared<nda_t>( make_vector_dims_t( get_ndat<T>().tn, vv.size() ) );
    std::copy( vv.begin(), vv.end(), static_cast< T * >(ret->rp_elems()) );
    return ret;
  }
  // make a no-data nda with the given dims
  inline p_nda_t make_dims_nda( dims_t const & dims ) { return make_shared<nda_t>( dims, (void *)0 ); }
  string get_scalar_c_const_str( nda_t const & nda );

  template< typename T > struct nda_T : public nda_t {
    nda_T( void ) { }
    nda_T( dims_t const & dims_ ) : nda_t( dims_ ) { check_tn(); }
    nda_T( p_nda_t const & nda_ ) : nda_t( *nda_ ) { assert_st( dims.valid() ); check_tn(); }
    void check_tn( void ) const {
      string const & type_tn = get_ndat<T>().tn; // FIXME: can the user trigger this error? use better err msg?
      if( dims.tn != type_tn ) { printf( "internal error: nda_T check_tn() failure: type_tn=%s dims.tn=%s\n", 
                                         str(type_tn).c_str(), str(dims.tn).c_str() ); }
    }
    T const * elems_ptr( void ) const { return reinterpret_cast<T const *>(d.get()); }
    T * elems_ptr( void ) { return reinterpret_cast<T *>(d.get()); }
    void copy_elems_with_reshape( nda_T< T > const & o ) {
      assert_st( elems_sz() == o.elems_sz() );
      for( uint32_t i = 0; i != elems_sz(); ++i ) { elems_ptr()[i] = o.elems_ptr()[i]; }
    }

    T & cm_at1( uint32_t const d0i ) { 
      uint32_t const ix = dims.cm_ix1( d0i ); return elems_ptr()[ix]; }
    T & cm_at2( uint32_t const d0i, uint32_t const d1i ) { 
      uint32_t const ix = dims.cm_ix2( d0i,d1i ); return elems_ptr()[ix]; }
    T & cm_at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      uint32_t const ix = dims.cm_ix3( d0i,d1i,d2i ); return elems_ptr()[ix]; }
    T & cm_at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      uint32_t const ix = dims.cm_ix4( d0i,d1i,d2i,d3i ); return elems_ptr()[ix]; }

    T & at( vect_uint32_t const & di ) { uint32_t const ix = dims.ix( di ); return elems_ptr()[ix]; }
    T & at( vect_uint32_t const & di, vect_uint32_t const & di2 ) { 
      uint32_t const ix = dims.ix_sum( di, di2 ); return elems_ptr()[ix]; }
    T & at1( uint32_t const d0i ) { 
      uint32_t const ix = dims.ix1( d0i ); return elems_ptr()[ix]; }
    T & at2( uint32_t const d0i, uint32_t const d1i ) { 
      uint32_t const ix = dims.ix2( d0i,d1i ); return elems_ptr()[ix]; }
    T & at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      uint32_t const ix = dims.ix3( d0i,d1i,d2i ); return elems_ptr()[ix]; }
    T & at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      uint32_t const ix = dims.ix4( d0i,d1i,d2i,d3i ); return elems_ptr()[ix]; }

    T const & cm_at1( uint32_t const d0i ) const { 
      uint32_t const ix = dims.cm_ix1( d0i ); return elems_ptr()[ix]; }
    T const & cm_at2( uint32_t const d0i, uint32_t const d1i ) const { 
      uint32_t const ix = dims.cm_ix2( d0i,d1i ); return elems_ptr()[ix]; }
    T const& cm_at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      uint32_t const ix = dims.cm_ix3( d0i,d1i,d2i ); return elems_ptr()[ix]; }
    T const& cm_at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      uint32_t const ix = dims.cm_ix4( d0i,d1i,d2i,d3i ); return elems_ptr()[ix]; }

    T const& at( vect_uint32_t const & di ) const { uint32_t const ix = dims.ix( di ); return elems_ptr()[ix]; }
    T const& at( vect_uint32_t const & di, vect_uint32_t const & di2 ) const { 
      uint32_t const ix = dims.ix_sum( di, di2 ); return elems_ptr()[ix]; }
    T const& at1( uint32_t const d0i ) const { 
      uint32_t const ix = dims.ix1( d0i ); return elems_ptr()[ix]; }
    T const& at2( uint32_t const d0i, uint32_t const d1i ) const { 
      uint32_t const ix = dims.ix2( d0i,d1i ); return elems_ptr()[ix]; }
    T const& at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) const { 
      uint32_t const ix = dims.ix3( d0i,d1i,d2i ); return elems_ptr()[ix]; }
    T const& at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) const { 
      uint32_t const ix = dims.ix4( d0i,d1i,d2i,d3i ); return elems_ptr()[ix]; }
  };
  template< typename STREAM, typename T > inline void bwrite( STREAM & out, nda_T<T> const & o ) { o.bwrite_( out ); }
  template< typename STREAM, typename T > inline void bread( STREAM & in, nda_T<T> & o ) { o.bread_( in ); o.check_tn(); }

  template< typename T > struct max_functor { 
    typedef T Result; 
    T operator()( T const & a, T const & b ) const { return std::max(a,b); } 
  };
  template< typename T > struct min_functor { 
    typedef T Result; 
    T operator()( T const & a, T const & b ) const { return std::min(a,b); } 
  };
  template< typename T > struct sum_functor { 
    typedef T Result; 
    T operator()( T const & a, T const & b ) const { return a+b; } 
  };
  
  template< typename VT, typename BF > typename BF::Result nda_reduce( VT const & vt, BF const & bf, 
								       typename BF::Result const & iv ) {
    typename BF::Result ret = iv;
    for( dims_iter_t di( vt.dims ) ; ; ) { ret = bf(ret,vt.at(di.di));  if( !di.next() ) { break; } }    
    return ret;
  }

  template< typename VT, typename BF > typename BF::Result nda_reduce( VT const & vt, BF const & bf, 
								       typename BF::Result const & iv,
								       dims_t const & b, dims_t const & e ) {
    typename BF::Result ret = iv;
    for( dims_iter_t di( b, e ) ; ; ) { ret = bf(ret,vt.at(di.di));  if( !di.next() ) { break; } }    
    return ret;
  }

  // yay template typedefs! what could go wrong? hoping for the best!
  template< typename T > using p_nda_T_t = shared_ptr< nda_T< T > >;
  template< typename T > using map_str_p_nda_T_t = map< string, p_nda_T_t<T> >;
  template< typename T > using p_map_str_p_nda_T_t = shared_ptr< map_str_p_nda_T_t<T> >;
  
  // since we can't direcly pass a container-of-derived as a container-of-base, we need to explicity create
  // containers-of-base as a workaround. c++!
  template< typename T > p_map_str_p_nda_t upcast_map_str_p_nda_T_t( p_map_str_p_nda_T_t<T> vs_T ) {
    p_map_str_p_nda_t vs = make_shared<map_str_p_nda_t>();
    for( typename map_str_p_nda_T_t<T>::const_iterator i = vs_T->begin(); i != vs_T->end(); ++i ) {
      must_insert( *vs, i->first, i->second );
    }
    return vs;
  }

  typedef nda_T< float > nda_float_t;

  typedef vector< nda_float_t > vect_nda_float_t; 
  typedef shared_ptr< nda_float_t > p_nda_float_t; 
  typedef vector< p_nda_float_t > vect_p_nda_float_t;
  typedef shared_ptr< vect_p_nda_float_t > p_vect_p_nda_float_t;

  typedef nda_T< double > nda_double_t;

  typedef vector< nda_double_t > vect_nda_double_t; 
  typedef shared_ptr< nda_double_t > p_nda_double_t; 
  typedef vector< p_nda_double_t > vect_p_nda_double_t;
  typedef map< string, p_nda_float_t > map_str_p_nda_float_t;
  typedef shared_ptr< map_str_p_nda_float_t > p_map_str_p_nda_float_t;
  typedef vector< p_map_str_p_nda_float_t > vect_p_map_str_p_nda_float_t;
  typedef map< string, p_vect_p_nda_float_t > map_str_p_vect_p_nda_float_t;
  typedef shared_ptr< map_str_p_vect_p_nda_float_t > p_map_str_p_vect_p_nda_float_t;

  typedef nda_T< uint8_t > nda_uint8_t;
  typedef nda_T< uint16_t > nda_uint16_t;
  typedef shared_ptr< nda_uint16_t > p_nda_uint16_t; 

  struct ssds_diff_per_type_t {
    virtual void cnt_diff_elems( void ) = 0;
    virtual void sum_squared_diffs( void ) = 0;
    virtual double get_rd_double( dims_iter_t const & di ) = 0;
    virtual string get_diff_str( dims_iter_t const & di ) = 0;
  };
  typedef shared_ptr< ssds_diff_per_type_t > p_ssds_diff_per_type_t; 

  // class to store info from sum_squared_diffs(o1,o2); str()-able for printing
  struct ssds_diff_t {
    p_nda_t o1;
    p_nda_t o2;

    p_ssds_diff_per_type_t pt;

    double ssds; // sum of squared differences
    double sds; // sum of differences
    double mad; // maximum absolute difference
    double mrd; // maximum relative difference
    uint32_t num_diff; // count of differing elements
    uint32_t sz; // number of elements
    double aad; // average absolute difference
    double ad; // average difference
    double sum1;
    double sum2;
    double avg1;
    double avg2;
    
    //ssds_diff_t( void ) { clear(); }
    void clear( void ) { ssds = 0; sds = 0; mad = 0; mrd = 0; num_diff = 0; sz = 0; aad = 0; ad = 0; 
      sum1 = 0; sum2 = 0; avg1 = 0; avg2 = 0; }
    bool has_nan( void ) const;
    ssds_diff_t( p_nda_t const & o1, p_nda_t const & o2 );
    string basic_str( void ) const;
  };
  std::ostream & operator <<(std::ostream & os, ssds_diff_t const & v);

  // create a digest of an NDA designed to be compact (like a hash) and insensitive (only) to small relative per-element
  // changes (like LSH)
  struct nda_digest_t; typedef shared_ptr< nda_digest_t > p_nda_digest_t;
  typedef vector< p_nda_digest_t > vect_p_nda_digest_t;
  typedef std::pair< string, p_nda_digest_t > pair_str_p_nda_digest_t;
  typedef vector< pair_str_p_nda_digest_t > vect_pair_str_p_nda_digest_t;

  struct nda_digest_t {
    // when comparing this nda_digest_t against 'itself' (the digest of the an nda generated in the same exact way),
    // we'd normally expect both the nda and thus the digest to be identical. however, sometimes we're using a
    // non-deterministic method of generating the nda, and this may introduce some (generall small) error. in such a
    // case, we will/must manually set self_cmp_mrd to an apropriate value when we create it.
    double self_cmp_mrd;
    nda_digest_t( void ) : self_cmp_mrd(0.0) { }
    // note: a minimal make/factor func could take only a tn string, but we don't need a minimal make/factory function currently
    static p_nda_digest_t make_from_nda( p_nda_t const & v, uint64_t const & seed ); // note: make minimally needs type str ...
    virtual void set_from_nda( p_nda_t const & v, uint64_t const & seed ) = 0;
    virtual string mrd_comp( p_nda_digest_t const & o, double const & mrd ) = 0;
    virtual string get_digest( void ) = 0;

    // IO functions and magic typedef to make below special global shared_ptr derived-as-base-class bread()/bwrite() enabled for this type
    typedef void has_static_bread_and_virt_bwrite;
    static p_nda_digest_t bread( std::istream & in );
    virtual void bwrite( std::ostream & out ) const = 0;
  };
  
  struct rt_exception : public std::exception
  {
    string err_msg;
    p_vect_rp_void bt;
    rt_exception( string const & err_msg_, p_vect_rp_void bt_ );
    virtual ~rt_exception() throw () {}
    virtual char const * what( void ) const throw();
    string what_and_stacktrace( void ) const;
    int get_ret_code( void ) const;
  };

  struct unsup_exception : public rt_exception {
    unsup_exception( string const & err_msg_, p_vect_rp_void bt_ ) : rt_exception( err_msg_, bt_ ) {}
  };

  // for any class to have nesi support, it must inherit ": public
  // virtual nesi", and *declare* get_cinfo() (which will be generated
  // in the corresponding .nesi_gen.cc file for the class declaration)
  struct cinfo_t;
  struct nesi { virtual cinfo_t const * get_cinfo( void ) const = 0; virtual ~nesi( void ) { } };
  typedef shared_ptr< nesi > p_nesi;

  struct read_req_t {
    virtual int get_fd( void ) = 0;
    virtual void on_readable( void ) = 0;
  };

  struct img_t; typedef shared_ptr< img_t > p_img_t; 
  struct img_proc_t { virtual void on_img( p_img_t const & img ) = 0; };

  void boda_dirs_init( void );
  std::string const & py_boda_dir( void ); 
  std::string const & py_boda_test_dir( void ); 

}

#endif // BODA_BASE_H

