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
#include"anno_util.H"
#include"rand_util.H"

namespace boda 
{
  struct display_test_t : virtual public nesi, public has_main_t // NESI(help="video display test",
			  // bases=["has_main_t"], type_id="display_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { 
      disp_win_t disp_win;
      disp_win.disp_setup( {{500,500},{200,200}} ); 
      p_vect_anno_t annos( new vect_anno_t );
      annos->push_back( anno_t{{{100,50},{200,250}}, rgba_to_pel(170,40,40), 0, "foo biz baz\nfoo bar\njiggy", rgba_to_pel(220,220,255) } );
      disp_win.update_img_annos( 0, annos );
      disp_win.update_img_annos( 1, annos );
      disp_win.update_disp_imgs();

      io_service_t & io = get_io( &disp_win );
      io.run();
    }
  };

  struct display_pil_t : virtual public nesi, public load_imgs_from_pascal_classes_t, public has_main_t // NESI(
			 // help="display PASCAL VOC list of images in video window",
			 // bases=["load_imgs_from_pascal_classes_t","has_main_t"], type_id="display_pil")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t rand_winds; //NESI(default=1,help="if set, display 1/2 image size random windows instead of full image")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    uint32_t cur_img_ix;
    boost::random::mt19937 gen;
    bool do_quit;
    display_pil_t( void ) : do_quit(0) { }
    void on_frame( error_code const & ec ) {
      if( do_quit ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) ); 
      if( cur_img_ix == all_imgs->size() ) { cur_img_ix = 0; }
      if( cur_img_ix < all_imgs->size() ) {
	p_img_t const & img = all_imgs->at(cur_img_ix);
	u32_pt_t src_pt, dest_pt;
	u32_pt_t copy_sz{u32_pt_t_const_max};
	if( rand_winds ) {
	  copy_sz = ceil_div( img->sz, {2,2} );
	  src_pt = random_pt( img->sz - copy_sz, gen );
	  dest_pt = random_pt( disp_imgs->at(0)->sz - copy_sz, gen );
	}
	img_copy_to_clip( img.get(), disp_imgs->at(0).get(), dest_pt, src_pt, copy_sz );
	++cur_img_ix;
      }
      disp_win.update_disp_imgs();
    }
    void on_quit( error_code const & ec ) { do_quit=1; frame_timer->cancel();  }

    virtual void main( nesi_init_arg_t * nia ) {
      load_all_imgs();

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
