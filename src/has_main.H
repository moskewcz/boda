#ifndef _HAS_MAIN_H_
#define _HAS_MAIN_H_

#include"boda_base.H"

namespace boda
{
  using std::string;
  struct lexp_name_val_map_t;
  typedef lexp_name_val_map_t nesi_init_arg_t;
  struct has_main_t : virtual public nesi // NESI(help="modes that can be run",is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="name of mode to run",req=1)
    filename_t boda_output_dir; //NESI(help="directory in which to place all outputs. will be created if non-existant.",req=1)
    void base_setup( void );
    virtual void main( nesi_init_arg_t * nia ) = 0;
  };
  typedef shared_ptr< has_main_t > p_has_main_t;
  typedef vector< p_has_main_t > vect_p_has_main_t;

  struct lexp_t;
  typedef shared_ptr< lexp_t > p_lexp_t;
  
  void create_and_run_has_main_t( p_lexp_t lexp );

}

#endif /* _HAS_MAIN_H_ */
