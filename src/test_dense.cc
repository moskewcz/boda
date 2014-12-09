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
#include"timers.H"

namespace boda {

  uint32_t const inmc = 123U+(117U<<8)+(104U<<16)+(255U<<24); // RGBA

  template< typename T >
  shared_ptr< nda_T< T > > copy_clip( shared_ptr< nda_T< T > > in, dims_t const & b, dims_t const & e ) {
    shared_ptr< nda_T< T > > ret( new nda_T< T > );
    dims_t ret_dims = e - b;
    ret->set_dims( ret_dims );
    dims_iter_t rdi( ret_dims );
    //printf( "dims=%s in->dims=%s\n", str(dims).c_str(), str(in->dims).c_str() );
    for( dims_iter_t di( b, e ) ; ; ) { 
      ret->at(rdi.di) = in->at(di.di); 
      if( !di.next() ) { break; } 
      assert_st( rdi.next() );
    }
    return ret;
  }
    

  struct test_dense_t : virtual public nesi, public has_main_t // NESI( help="test dense vs. sparse CNN eval",
			// bases=["has_main_t"], type_id="test_dense")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test_dense.txt",help="output: text summary of differences between dense and sparse feature computation.")
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

    p_conv_pipe_t conv_pipe_dense;
    p_vect_conv_io_t conv_ios_dense;

    void get_pipe_and_ios( p_run_cnet_t const & run_cnet, p_conv_pipe_t & conv_pipe, p_vect_conv_io_t & conv_ios ) {
      conv_pipe = run_cnet->get_pipe();
      conv_pipe->dump_pipe( *out );
      conv_ios = conv_pipe->calc_sizes_forward( run_cnet->in_sz, 0 ); 
      conv_pipe->dump_ios( *out, conv_ios );
    }
    
    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      imgs->load_all_imgs();
      run_cnet->setup_cnet(); 
      for( vect_p_img_t::const_iterator i = imgs->all_imgs->begin(); i != imgs->all_imgs->end(); ++i ) {
	run_cnet_dense->in_sz.max_eq( (*i)->sz );
      }
      run_cnet_dense->setup_cnet();

      in_img = make_p_img_t( run_cnet->in_sz );
      in_img_dense = make_p_img_t( run_cnet_dense->in_sz );

      get_pipe_and_ios( run_cnet, conv_pipe, conv_ios );
      get_pipe_and_ios( run_cnet_dense, conv_pipe_dense, conv_ios_dense );
      
      boost::random::mt19937 gen;

      uint32_t tot_wins = 0;
      for( vect_p_img_t::const_iterator i = imgs->all_imgs->begin(); i != imgs->all_imgs->end(); ++i ) {
	if( !(*i)->sz.both_dims_ge( run_cnet->in_sz ) ) { continue; } // img too small to sample. assert? warn?
	(*out) << strprintf( "(*i)->sz=%s\n", str((*i)->sz).c_str() );
	// run net on entire input image
	in_img_dense->fill_with_pel( inmc );
	img_copy_to_clip( (*i).get(), in_img_dense.get() );
	subtract_mean_and_copy_img_to_batch( run_cnet_dense->in_batch, 0, in_img_dense );
	p_nda_float_t out_batch_dense;
	{
	  timer_t t1("dense_cnn");
	  out_batch_dense = run_cnet_dense->run_one_blob_in_one_blob_out();
	}
      
	for( uint32_t wix = 0; wix != wins_per_image; ++wix ) {
	  u32_pt_t const samp_nc_max = (*i)->sz - run_cnet->in_sz;
	  u32_pt_t const samp_nc = random_pt( samp_nc_max, gen );
	  ++tot_wins;
	  comp_win( out_batch_dense, (*i), samp_nc );
	}
      }
    }
    void comp_win( p_nda_float_t out_batch_dense, p_img_t const & img, u32_pt_t const & nc ) {
      dims_t const & obd_dense = out_batch_dense->dims;
      assert( obd_dense.sz() == 4 );
      assert( obd_dense.dims(0) == 1 ); // one image
      //printf( "nc=%s\n", str(nc).c_str() );   
      u32_box_t in_box{ nc, nc + run_cnet->in_sz };
      conv_support_info_t const & ol_csi_dense = conv_pipe_dense->conv_sis.back();
      i32_box_t feat_box_dense;
      in_box_to_out_box( feat_box_dense, in_box, cm_valid, ol_csi_dense );

      u32_pt_t const & feat_sz = conv_ios->back().sz;
      if( i32_to_u32(feat_box_dense.sz()) != feat_sz ) { return; }
      
      (*out) << strprintf( "feat_box_dense=%s\n", str(feat_box_dense).c_str() );
      // run net on just sample area
      img_copy_to_clip( img.get(), in_img.get(), {}, nc );
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, in_img );
      p_nda_float_t out_batch;
      {
	timer_t t1("sparse_cnn");
	out_batch = run_cnet->run_one_blob_in_one_blob_out();
	dims_t const & obd = out_batch->dims;
	assert( obd.sz() == 4 );
	assert( obd.dims(0) == 1 ); // one image
	assert( obd.dims(1) == obd_dense.dims(1) ); // same chans

	dims_t b(4);
	dims_t e = obd_dense;
	b.dims(2) = feat_box_dense.p[0].d[1];
	b.dims(3) = feat_box_dense.p[0].d[0];
	e.dims(2) = b.dims(2) + feat_sz.d[1];
	e.dims(3) = b.dims(3) + feat_sz.d[0];
	p_nda_float_t from_dense = copy_clip( out_batch_dense, b, e );
	(*out) << strprintf( "ssds_str(from_dense,out_batch)=%s\n", str(ssds_str(from_dense,out_batch)).c_str() );
#if 0	
	for( uint32_t c = 0; c < obd.dims(1); ++c ) {
	  for( uint32_t y = 0; y < feat_sz.d[1]; ++y ) {
	    for( uint32_t x = 0; x < feat_sz.d[0]; ++x ) {
	      float const v = out_batch->at4(0,c,y,x);
	      float const vd = out_batch_dense->at4(0,c,y+feat_box_dense.p[0].d[1],x+feat_box_dense.p[0].d[0]);
	      printf( "v=%s vd=%s\n", str(v).c_str(), str(vd).c_str() );
	    }
	  }
	}
#endif
      }
    }
  };

#include"gen/test_dense.cc.nesi_gen.cc"
  
}
