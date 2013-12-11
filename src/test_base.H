#ifndef _TEST_BASE_H_
#define _TEST_BASE_H_

#include"str_util.H"

namespace boda 
{

#if 0 // FIXME: currently unused
  // test interface
  struct boda_test_t {
    virtual void run( void );
    virtual void post( void );
    virtual char * exp_error( void ); // null if no error expected. valid only after run.
  };
  typedef vector< boda_test_t > vect_boda_test_t; 
  typedef shared_ptr< boda_test_t > p_boda_test_t; 
  typedef vector< p_boda_test_t > vect_p_boda_test_t;
  // test set interface
  struct test_set_t {
    uint32_t num_tests( void );

  };
  struct REBASE_lexp_test_t : public boda_test_t {

  };
#endif

  struct test_run_t {
    uint32_t tix;
    uint32_t num_fail;

    virtual void test_print( void ) = 0;
    void test_fail( void ) {
      ++num_fail;
      test_print();
    }
    void test_fail_no_err( string const & msg )  {
      test_fail();
      printf( "test failed: missing expected error:\n  %s\n", str(msg).c_str() );
    }
    void test_fail_err( string const & msg )  {
      test_fail();
      printf( "test failed: expected no error, but error was:\n  %s\n", str(msg).c_str() );
    }
    void test_fail_wrong_res( string const & msg )  {
      test_fail();
      printf( "test failed: wrong result:\n  %s\n", str(msg).c_str() );
    }
    void test_fail_wrong_err( string const & msg )  {
      test_fail();
      printf( "test failed: wrong error, got:\n%s\n", str(msg).c_str() );
    }
  };

}
#endif /* _TEST_BASE_H_ */
