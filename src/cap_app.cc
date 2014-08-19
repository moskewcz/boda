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
#include"cap_util.H"
#include"caffeif.H"

namespace boda 
{
  struct capture_classify_t : virtual public nesi, public has_main_t // NESI(help="cnet classifaction from video capture",
			      // bases=["has_main_t"], type_id="capture_classify")
			    , public img_proc_t
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_capture_t capture; //NESI(default="()",help="capture from camera options")    
    p_cnet_predict_t cnet_predict; //NESI(default="()",help="cnet running options")    
    virtual void main( nesi_init_arg_t * nia ) { cnet_predict->setup_predict(); capture->cap_loop( this ); }
    virtual void on_img( p_img_t const & img ) { cnet_predict->do_predict( img ); }
  };

  struct capture_feats_t : virtual public nesi, public has_main_t // NESI(help="cnet classifaction from video capture",
			   // bases=["has_main_t"], type_id="capture_feats")
			 , public img_proc_t
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_capture_t capture; //NESI(default="()",help="capture from camera options")    
    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt,out_layer_name=conv5)",help="cnet running options")
    virtual void main( nesi_init_arg_t * nia ) { 
      run_cnet->in_sz = capture->cap_res;
      run_cnet->setup_cnet(); 
      capture->cap_loop( this ); 
    }
    virtual void on_img( p_img_t const & img ) { 
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, img );
      p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
    }
  };


#include"gen/cap_app.cc.nesi_gen.cc"

}
