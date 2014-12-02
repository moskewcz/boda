// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"caffeif.H"
#include"conv_util.H"

namespace boda {

  struct test_dense_t : virtual public nesi, public has_main_t // NESI(help="test dense vs. sparse CNN eval",
			// bases=["has_main_t"], type_id="test_dense")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")

    p_vect_p_img_t all_imgs;

    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt,out_layer_name=conv3)",help="cnet running options")
    p_img_t in_img;
    p_img_t feat_img;

    p_conv_pipe_t conv_pipe;
    p_vect_conv_io_t conv_ios;

    virtual void main( nesi_init_arg_t * nia ) {
      p_vect_string classes = readlines_fn( pascal_classes_fn );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	bool const is_first_class = (i == (*classes).begin());
	read_pascal_image_list_file( img_db, filename_t_printf( pil_fn, (*i).c_str() ), 
				     true && is_first_class, !is_first_class );
      }
      all_imgs.reset( new vect_p_img_t );
      img_db_get_all_loaded_imgs( all_imgs, img_db );


      run_cnet->setup_cnet(); 
      in_img.reset( new img_t );
      in_img->set_sz_and_alloc_pels( run_cnet->in_sz );

      conv_pipe = run_cnet->get_pipe();
      conv_pipe->dump_pipe( std::cout );
      conv_ios = conv_pipe->calc_sizes_forward( run_cnet->in_sz, 0 ); 
      conv_pipe->dump_ios( std::cout, conv_ios );
      feat_img.reset( new img_t );
      u32_pt_t const feat_img_sz = run_cnet->get_one_blob_img_out_sz();
      feat_img->set_sz_and_alloc_pels( feat_img_sz );

    }
  };

#include"gen/test_dense.cc.nesi_gen.cc"
  
}
