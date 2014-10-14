// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"disp_util.H"
#include"cap_util.H"

#include"asio_util.H"

namespace boda 
{
  struct display_test_t : virtual public nesi, public has_main_t // NESI(help="video display test",
			  // bases=["has_main_t"], type_id="display_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { 
      disp_win_t disp_win;
      disp_win.disp_setup( {{100,100}} ); 
      io_service_t & io = get_io( &disp_win );
      io.run();
    }
  };

  struct display_pil_t : virtual public nesi, public has_main_t // NESI(help="display PASCAL VOC list of images in video window",
		      // bases=["has_main_t"], type_id="display_pil")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")

    disp_win_t disp_win;
    p_vect_p_img_t all_imgs;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    uint32_t cur_img_ix;

    void on_frame( error_code const & ec ) {
      if( ec == errc::operation_canceled ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) ); 
      if( cur_img_ix == all_imgs->size() ) { cur_img_ix = 0; }
      if( cur_img_ix < all_imgs->size() ) {
	img_copy_to_clip( all_imgs->at(cur_img_ix).get(), disp_imgs->at(0).get(), 0, 0 );
	++cur_img_ix;
      }
      disp_win.update_disp_imgs();
    }
    void on_quit( error_code const & ec ) { frame_timer->cancel(); }

    virtual void main( nesi_init_arg_t * nia ) {
      p_vect_string classes = readlines_fn( pascal_classes_fn );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	bool const is_first_class = (i == (*classes).begin());
	read_pascal_image_list_file( img_db, filename_t_printf( pil_fn, (*i).c_str() ), 
				     true && is_first_class, !is_first_class );
      }
      all_imgs.reset( new vect_p_img_t );
      img_db_get_all_loaded_imgs( all_imgs, img_db );

      disp_imgs = disp_win.disp_setup( {{640,480}} );
      
      cur_img_ix = 0;
      io_service_t & io = get_io( &disp_win );
      frame_timer.reset( new deadline_timer_t( io ) );
      frame_dur = microseconds( 1000 * 1000 / fps );
      frame_timer->expires_from_now( time_duration() );
      frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) );
      register_quit_handler( disp_win, &display_pil_t::on_quit, this );
      io.run();
    }

  };

#include"gen/disp_app.cc.nesi_gen.cc"

}
