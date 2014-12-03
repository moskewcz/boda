// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"caffeif.H"
#include"conv_util.H"
#include"rand_util.H"

namespace boda {

  struct test_dense_t : virtual public nesi, public has_main_t // NESI( help="test dense vs. sparse CNN eval",
			// bases=["has_main_t"], type_id="test_dense")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_load_imgs_from_pascal_classes_t imgs;//NESI(default="()")
    string model_name; //NESI(default="nin_imagenet_nopad",help="name of model")
    p_run_cnet_t run_cnet; //NESI(default="(in_sz=227 227,ptt_fn=%(models_dir)/%(model_name)/deploy.prototxt.boda
                           // ,trained_fn=%(models_dir)/%(model_name)/best.caffemodel
                           // ,out_layer_name=relu12)",help="CNN model params")
    p_run_cnet_t run_cnet_dense; //NESI(default="(in_sz=0 0,ptt_fn=%(models_dir)/%(model_name)/deploy.prototxt.boda" 
                                 // ",trained_fn=%(models_dir)/%(model_name)/best.caffemodel"
                                 // ",out_layer_name=relu12)",help="CNN model params")
    uint32_t wins_per_image; //NESI(default="1",help="number of random windows per image to test")

    p_img_t in_img;
    p_img_t in_img_dense;

    p_conv_pipe_t conv_pipe;
    p_vect_conv_io_t conv_ios;

    
    virtual void main( nesi_init_arg_t * nia ) {
      imgs->load_all_imgs();
      run_cnet->setup_cnet(); 
      for( vect_p_img_t::const_iterator i = imgs->all_imgs->begin(); i != imgs->all_imgs->end(); ++i ) {
	run_cnet_dense->in_sz.max_eq( (*i)->sz );
      }
      run_cnet_dense->setup_cnet();

      in_img = make_p_img_t( run_cnet->in_sz );
      in_img_dense = make_p_img_t( run_cnet_dense->in_sz );

      conv_pipe = run_cnet->get_pipe();
      //conv_pipe->dump_pipe( std::cout );
      conv_ios = conv_pipe->calc_sizes_forward( run_cnet->in_sz, 0 ); 
      //conv_pipe->dump_ios( std::cout, conv_ios );
      
      boost::random::mt19937 gen;

      uint32_t tot_wins = 0;
      for( vect_p_img_t::const_iterator i = imgs->all_imgs->begin(); i != imgs->all_imgs->end(); ++i ) {
	if( !(*i)->sz.both_dims_ge( run_cnet->in_sz ) ) { continue; } // img too small to sample. assert? warn?
	u32_pt_t const samp_sz = (*i)->sz - run_cnet->in_sz;
	u32_pt_t const samp_nc = random_pt( samp_sz, gen );
	++tot_wins;
	comp_win( (*i), samp_nc );
      }
    }
    void comp_win( p_img_t const & img, u32_pt_t const & nc ) {
      printf( "nc=%s\n", str(nc).c_str() );      
      // run net on just sample area
      img_copy_to_clip( img.get(), in_img.get(), {}, nc );
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, in_img );
      p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();

      uint32_t const inmc = 123U+(117U<<8)+(104U<<16)+(255U<<24); // RGBA

      // run net on entire input image
      in_img_dense->fill_with_pel( inmc );
      img_copy_to_clip( img.get(), in_img_dense.get() );
      subtract_mean_and_copy_img_to_batch( run_cnet_dense->in_batch, 0, in_img_dense );
      p_nda_float_t out_batch_dense = run_cnet_dense->run_one_blob_in_one_blob_out();

    }
  };

#include"gen/test_dense.cc.nesi_gen.cc"
  
}
