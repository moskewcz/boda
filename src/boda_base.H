// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef BODA_BASE_H
#define BODA_BASE_H

#include<cassert>
#include<boost/shared_ptr.hpp>
#include<boost/make_shared.hpp>
#include<fstream>
#include<string>
#include<vector>
#include<map>
#include<set>
#include<exception>
#include<poll.h>
#include<boost/integer_traits.hpp>

# define assert_st(expr)							\
  ((expr)								\
   ? __ASSERT_VOID_CAST (0)						\
   : __assert_fail ((std::string(" failed:\n") + __STRING(expr)+\
		     "\n "+stacktrace_str( get_backtrace(), 0 )).c_str(),	\
		    __FILE__, __LINE__, __ASSERT_FUNCTION))


namespace boost {
  namespace iostreams {
    struct mapped_file;
    struct mapped_file_source;
    struct mapped_file_sink;
  }
}
namespace boda
{
  struct zi_uint32_t { uint32_t v; zi_uint32_t( void ) : v(0) { } };
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

  using boost::shared_ptr;
  using boost::make_shared;
  template< typename T > struct null_deleter { void operator()( T * const ) const {}; };
  using boost::dynamic_pointer_cast;
  using boost::iostreams::mapped_file;
  using boost::iostreams::mapped_file_source;
  using boost::iostreams::mapped_file_sink;
  using std::string;
  using std::vector;
  using std::map;
  using std::set;

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

  template< typename T > void must_insert( T & kvm, typename T::key_type const & k ) {
    bool const did_ins = kvm.insert( k ).second;
    assert_st( did_ins );
  }		      
  template< typename T > void must_insert( T & kvm, typename T::key_type const & k, typename T::mapped_type const & v ) {
    bool const did_ins = kvm.insert( std::make_pair( k, v ) ).second;
    assert_st( did_ins );
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

  template< typename T > typename T::mapped_type const & get( T const & kvm, typename T::key_type const & k, typename T::mapped_type const & dv ) {
    typename T::const_iterator i = kvm.find( k );
    if( i == kvm.end() ) { return dv; }
    return i->second;
  }

  // NESI types
  struct filename_t { string in; string exp; };
  typedef vector< filename_t > vect_filename_t; 
  typedef shared_ptr< filename_t > p_filename_t; 
  typedef vector< p_filename_t > vect_p_filename_t;

  // typedef uint32_t nesi_bool // TODO

  typedef vector< string > vect_string;
  typedef shared_ptr< string > p_string;
  typedef shared_ptr< vect_string > p_vect_string;
  vect_rp_char get_vect_rp_char( vect_string const & v );
  vect_rp_const_char get_vect_rp_const_char( vect_string const & v );
  typedef vector< std::pair< string, string > > vect_pair_str_str;

  typedef set< string > set_string;
  typedef set< uint32_t > set_uint32_t;

  typedef map< string, uint32_t > map_str_uint32_t;
  typedef map< string, zi_uint32_t > map_str_ziu32_t;
  typedef map< string, double > map_str_double;
  typedef map< string, string > map_str_str;
  typedef map< string, vect_string > map_str_vect_string;

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
  typedef shared_ptr< std::ofstream > p_ofstream;
  bool ifs_getline( std::string const &fn, p_ifstream in, std::string & line );
  p_ofstream ofs_open( filename_t const & fn );
  p_ofstream ofs_open( string const & fn );
  p_string read_whole_fn( filename_t const & fn );
  p_string read_whole_fn( string const & fn );
  void write_whole_fn( filename_t const & fn, std::string const & data );
  void write_whole_fn( string const & fn, std::string const & data );

  p_vect_string readlines_fn( filename_t const & fn );
  p_vect_string readlines_fn( string const & fn );

  void neg_one_fail( int const & ret, char const * const func_name );
  void fork_and_exec_self( vect_string const & args );
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
  typedef vector< float > vect_float; 
  typedef shared_ptr< float > p_float;
  typedef vector< double > vect_double; 
  typedef shared_ptr< double > p_double; 

  p_uint8_t make_mmap_shared_p_uint8_t( int const fd, size_t const length, off_t const offset );
  void remap_mmap_shared_p_uint8_t( p_uint8_t &p, size_t const new_length );
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
    hunk_T< T > clone( void ) const { // clone will have newly allocated memory with a copy of the data
      hunk_T<T> ret( sz ); 
      for( uint32_t ix = 0; ix < sz; ++ix ) { ret[ix] = (*this)[ix]; }
      return ret;
    }
  };

  // note: difference is o2 - o1, i.e. the delta/diff to get to o2 from o1. mad == max absolute diff
  template< typename RT, typename DT >
  void sum_squared_diffs( RT & ssd, RT & sd, RT & mad, DT const & o1, DT const & o2 ) {
    assert_st( o1.sz == o2.sz );
    for( uint32_t i = 0; i < o1.sz; ++i ) { 
      RT const d = RT(o2[i])-RT(o1[i]);
      sd += d; ssd += d*d; max_eq(mad,(d<0)?-d:d);
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
#endif
  template< typename T, typename check_T<typename T::has_bread_bwrite>::int_ = 0 > 
  void bwrite_bytes( T & out, char const * const & d, size_t const & sz );
  template< typename T, typename check_T<typename T::has_bread_bwrite>::int_ = 0  >
  void bread_bytes( T & in, char * const & d, size_t const & sz );


  template< typename STREAM, typename T > inline void bwrite( STREAM & out, T const & o ) { 
    bwrite_bytes( out, (char *)&o, sizeof(o) ); }
  template< typename STREAM, typename T > inline void bread( STREAM & in, T & o ) { 
    bread_bytes( in, (char *)&o, sizeof(o) ); }
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

  template< typename STREAM, typename T > inline void bwrite( STREAM & out, shared_ptr< T > const & o ) { 
    bwrite( out, *o ); }
  template< typename STREAM, typename T > inline void bread( STREAM & in, shared_ptr< T > & o ) { 
    o.reset( new T );
    bread( in, *o );
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

  template< typename STREAM >
  inline void bwrite_id( STREAM & out, string const & o ) { bwrite( out, string("id") ); bwrite( out, o ); }

  template< typename STREAM, typename T > inline void bwrite( STREAM & out, hunk_T< T > const & o ) { 
    bwrite( out, o.sz );
    bwrite_bytes( out, (char *)o.p.get(), o.sz*sizeof(T) );
  }
  template< typename STREAM, typename T > inline void bread( STREAM & in, hunk_T< T > & o ) { 
    bread( in, o.sz );
    o.p = ma_p_T<T>( o.sz );
    bread_bytes( in, (char *)o.p.get(), o.sz*sizeof(T) );
  }

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

  // row-major dimensions holder
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
    // string type;  // i.e. "float", "uint32_t", "int8_t", etc ...


    //hunk_uint32_t dss; // dims and strides, strides first, in T units (i.e. not neccessarily bytes). can be set manually.
    dims_t clone( void ) const { return *this; }

    // note: == and < compare: dims *and* strides *and* names
    bool operator == ( dims_t const & o ) const { return ((vect_dim_t const &)(*this)) == ((vect_dim_t const &)(o)); } 
    bool operator < ( dims_t const & o ) const { return ((vect_dim_t const &)(*this)) < ((vect_dim_t const &)(o)); } 

    // like ==, but treats zero/empty sz/stride/names as wild in the passed template/'o' dims_t (NOT in the *this dims_t)
    bool matches_template( dims_t const & o ) const { // note: o is the template, not *this
      if( sz() != o.sz() ) { return 0; }
      for( uint32_t i = 0; i != sz(); ++i ) { if( !at(i).matches_template( o.at(i) ) ) { return 0; } }
      return 1;
    }
    bool has_sz_and_stride_and_name( void ) const { 
      for( uint32_t i = 0; i != sz(); ++i ) { if( !at(i).has_sz_and_stride_and_name() ) { return 0; } }
      return 1;
    }
    bool has_name( void ) const { 
      for( uint32_t i = 0; i != sz(); ++i ) { if( !at(i).has_name() ) { return 0; } }
      return 1;
    }

    bool fits_in( dims_t const & o ) const { 
      if( sz() != o.sz() ) { return 0; }
      for( uint32_t ix = 0; ix != sz(); ++ix ) { if( dims(ix) > o.dims(ix) ) { return 0; } }
      return 1;
    }
    dims_t( void ) { }
    explicit dims_t( uint32_t const num_dims ) { resize_and_zero( num_dims ); }
    dims_t( vect_uint32_t const & dims_, bool const & calc_no_pad_strides = 0 ) { init( dims_, 0, calc_no_pad_strides ); }
    dims_t( vect_uint32_t const & dims_, vect_string const & names_, bool const & calc_no_pad_strides = 0 ) { 
      init( dims_, &names_, calc_no_pad_strides ); }
    void init( vect_uint32_t const & dims_, vect_string const * const names_, bool const & calc_no_pad_strides = 0 ) { 
      resize_and_zero( dims_.size() ); 
      for( uint32_t i = 0; i != dims_.size(); ++i ) { dims(i) = dims_[i]; }
      if( calc_no_pad_strides ) { calc_strides(); }
      if( names_ ) {
	assert_st( names_->size() == size() );
	for( uint32_t i = 0; i != size(); ++i ) { names(i) = names_->at(i); }
      }
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
      dims_t ret( sz() );
      for( uint32_t ix = 0; ix != sz(); ++ix ) { ret.dims(ix) = dims(ix) - o.dims(ix); } 
      return ret;
    }
    uint32_t sz( void ) const { return size(); }

    uint32_t & dims( uint32_t const ix ) { return at(ix).sz; }
    string & names( uint32_t const ix ) { return at(ix).name; }
    uint32_t & strides( uint32_t const ix ) { return at(ix).stride; }

    uint32_t const & dims( uint32_t const ix ) const { return at(ix).sz; }
    string const & names( uint32_t const ix ) const { return at(ix).name; }
    uint32_t const & strides( uint32_t const ix ) const { return at(ix).stride; }

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

    uint32_t calc_strides( void ) { // assuming no padding, i.e. unit stride for fast dim and so on
      strides(sz()-1) = 1;
      for( uint32_t d = 1; d < sz(); ++d ) { strides(sz()-d-1) = strides(sz()-d)*dims(sz()-d); }
      return strides(0)*dims(0); // i.e. tot_elems, strides(-1), size of all data, used to bounds check
    }

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

  inline std::ostream & operator << ( std::ostream & out, dims_t const & o ) {
    out << "RM_DIMS=";
    for( uint32_t ix = 0; ix != o.sz(); ++ix ) { out << (ix?"*":"") << o.dims(ix); }
    if( o.has_any_strides() ) {
      out << " STRIDES=";
      for( uint32_t ix = 0; ix != o.sz(); ++ix ) { out << (ix?",":"") << o.strides(ix); }
    }
    if( o.has_any_names() ) {
      out << " NAMES=";
      for( uint32_t ix = 0; ix != o.sz(); ++ix ) { out << (ix?",":"") << o.names(ix); }
    }
    return out;
  }


  template< typename STREAM > inline void bwrite( STREAM & out, dim_t const & o ) { 
    bwrite( out, o.sz ); bwrite( out, o.stride ); bwrite( out, o.name ); 
  }
  template< typename STREAM > inline void bread( STREAM & in, dim_t & o ) {
    bread( in, o.sz ); bread( in, o.stride ); bread( in, o.name );  
  }

  // for dims_t, let bread/bwrite template magic know to use standard vector bread/bwrite
  template< typename STREAM > inline void bwrite( STREAM & out, dims_t const & o ) { bwrite( out, (vect_dim_t const &)o ); }
  template< typename STREAM > inline void bread( STREAM & in, dims_t & o ) { bread( in, (vect_dim_t &)o ); }

  template< typename T > struct nda_T {
    nda_T( void ) { }
    nda_T( dims_t const & dims_ ) { set_dims( dims_ ); }
    hunk_T< T > elems; 
    dims_t dims;
  private:
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
  public:

    T & cm_at1( uint32_t const d0i ) { 
      uint32_t const ix = dims.cm_ix1( d0i ); return elems[ix]; }
    T & cm_at2( uint32_t const d0i, uint32_t const d1i ) { 
      uint32_t const ix = dims.cm_ix2( d0i,d1i ); return elems[ix]; }
    T & cm_at3( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i ) { 
      uint32_t const ix = dims.cm_ix3( d0i,d1i,d2i ); return elems[ix]; }
    T & cm_at4( uint32_t const d0i, uint32_t const d1i, uint32_t const d2i, uint32_t const d3i ) { 
      uint32_t const ix = dims.cm_ix4( d0i,d1i,d2i,d3i ); return elems[ix]; }

    T & at( vect_uint32_t const & di ) { uint32_t const ix = dims.ix( di ); return elems[ix]; }
    T & at( vect_uint32_t const & di, vect_uint32_t const & di2 ) { 
      uint32_t const ix = dims.ix_sum( di, di2 ); return elems[ix]; }
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
    T const& at( vect_uint32_t const & di, vect_uint32_t const & di2 ) const { 
      uint32_t const ix = dims.ix_sum( di, di2 ); return elems[ix]; }
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
    if( o.dims.sz() > 1 ) {
      // print rows by iterating over dim with fastest/smallest stride (using column major indexing, of course)
      for( uint32_t rix = 0; o.dims.cm_ix2( 0, rix ) != o.elems.sz; ++rix ) {
	out << "row=" << rix << ":";
	for( uint32_t cix = 0; cix != o.dims.cm_ix2( 0, 1 ); ++cix ) {
	  out << " " << o.cm_at2( cix, rix );
	}
	out << std::endl;
      }
    } else { // 1D case
      for( uint32_t ix = 0; ix != o.elems.sz; ++ix ) {
	out << " " << o.elems[ix];
      }
    }    

    return out;
  }

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



  template< typename STREAM, typename T > inline void bwrite( STREAM & out, nda_T<T> const & o ) { 
    bwrite(out,o.dims); bwrite(out,o.elems); }
  template< typename STREAM, typename T > inline void bread( STREAM & in, nda_T<T> & o ) { 
    bread(in,o.dims); bread(in,o.elems); }
  
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
  typedef map< string, p_vect_p_nda_float_t > map_str_p_vect_p_nda_float_t;
  typedef shared_ptr< map_str_p_vect_p_nda_float_t > p_map_str_p_vect_p_nda_float_t;

  // class to store info from sum_squared_diffs(o1,o2); str()-able for printing
  struct ssds_diff_t {
    double ssds;
    double sds;
    double mad;
    uint32_t num_diff;
    uint32_t sz;
    ssds_diff_t( void ) : ssds(0), sds(0), mad(0), num_diff(0), sz(0) { }
    bool has_nan( void ) const;
    template< typename T > ssds_diff_t( T const & o1, T const & o2 );
    string str( void ) const;
  };
  std::ostream & operator <<(std::ostream & os, ssds_diff_t const & v);

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

  // for any class to have nesi support, it must inherit ": public
  // virtual nesi", and *declare* get_cinfo() (which will be generated
  // in the corresponding .nesi_gen.cc file for the class declaration)
  struct cinfo_t;
  struct nesi { virtual cinfo_t const * get_cinfo( void ) const = 0; };
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

