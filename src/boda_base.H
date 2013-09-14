#ifndef BODA_BASE_H
#define BODA_BASE_H

#include<cassert>
#include<boost/shared_ptr.hpp>
#include<fstream>
#include<string>
#include<vector>
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
  using boost::shared_ptr;
  typedef shared_ptr< std::ifstream > p_ifstream;
  p_ifstream ifs_open( std::string const & fn );

  typedef std::vector< void * > vect_rp_void;
  typedef shared_ptr< vect_rp_void > p_vect_rp_void;
  p_vect_rp_void get_backtrace( void );
  std::string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames );
  void ensure_is_dir( const boost::filesystem3::path & p );
  void ensure_is_regular_file( const boost::filesystem3::path & p );

  struct rt_exception : public std::exception
  {
    std::string err_msg;
    p_vect_rp_void bt;
    std::string what_str;
    rt_exception( std::string const & err_msg_, p_vect_rp_void bt_ );
    virtual ~rt_exception() throw () {}
    virtual char const * what( void ) const throw();
    std::string what_and_stacktrace( void ) const;
    int get_ret_code( void ) const;
  };
  // raise runtime error. captures stack and raises exception with
  // given string as error message. as a perf. compromise, while the
  // top frames of the stack are always captured, they are not
  // processed into a full trace at throw time.
  void rt_err( std::string const & err_msg ) __attribute__((noreturn));
}

#endif // BODA_BASE_H

