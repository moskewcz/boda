// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"disp_util.H"

namespace boda 
{
  using namespace boost;

  struct cap_skel_t : virtual public nesi, public has_main_t // NESI(help="video capture skeleton",
		      // bases=["has_main_t"], type_id="cap_skel")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_100/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_100/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")

    virtual void main( nesi_init_arg_t * nia ) { 
      p_vect_string classes = readlines_fn( pascal_classes_fn );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	bool const is_first_class = (i == (*classes).begin());
	read_pascal_image_list_file( img_db, filename_t_printf( pil_fn, (*i).c_str() ), true && is_first_class , !is_first_class );
      }
      p_vect_p_img_t disp_imgs = img_db_get_all_loaded_imgs( img_db );
      
      disp_win_t disp_win;
      disp_win.disp_skel( *disp_imgs ); 

    }
  };

#include"gen/cap_util.cc.nesi_gen.cc"

}
