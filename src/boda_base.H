#ifndef BODA_BASE_H
#define BODA_BASE_H

#include<cassert>
#include<boost/shared_ptr.hpp>
#include<fstream>
#include<string>
#include<vector>
#include<map>
#include<exception>
#include<boost/integer_traits.hpp>

# define assert_st(expr)							\
  ((expr)								\
   ? __ASSERT_VOID_CAST (0)						\
   : __assert_fail ((std::string(" failed:\n") + __STRING(expr)+\
		     "\n "+stacktrace_str( get_backtrace(), 0 )).c_str(),	\
		    __FILE__, __LINE__, __ASSERT_FUNCTION))
namespace boost 
{
  namespace filesystem3
  {
    struct path;
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
  using std::string;
  using std::vector;
  using std::map;

  typedef vector< string > vect_string;

  typedef shared_ptr< void > p_void;
  typedef shared_ptr< std::ifstream > p_ifstream;
  p_ifstream ifs_open( string const & fn );
  typedef shared_ptr< std::ofstream > p_ofstream;
  p_ofstream ofs_open( string const & fn );

  typedef std::vector< void * > vect_rp_void;
  typedef shared_ptr< vect_rp_void > p_vect_rp_void;
  p_vect_rp_void get_backtrace( void );
  string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames );
  void ensure_is_dir( string const & fn );
  void ensure_is_dir( const boost::filesystem3::path & p );
  void ensure_is_regular_file( string const & fn );
  void ensure_is_regular_file( const boost::filesystem3::path & p );

  typedef std::vector< uint8_t > vect_uint8_t;  
  typedef shared_ptr< uint8_t > p_uint8_t;  
  p_uint8_t ma_p_uint8_t( uint32_t const sz, uint32_t const a ); // ma_p == make aligned shared_ptr

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

