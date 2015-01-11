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
  shared_ptr< nda_T< T > > copy_clip( shared_ptr< nda_T< T > > const & in, dims_t const & b, dims_t const & e ) {
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

  p_nda_float_t feats_copy_clip( p_nda_float_t const & in, i32_box_t const & cbox ) {
    dims_t b(4);
    dims_t e = in->dims;
    b.dims(2) = cbox.p[0].d[1];
    e.dims(2) = cbox.p[1].d[1];
    b.dims(3) = cbox.p[0].d[0];
    e.dims(3) = cbox.p[1].d[0];
    return copy_clip( in, b, e );
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

    void dump_pipe_and_ios( p_run_cnet_t const & rc ) {
      rc->conv_pipe->dump_pipe( *out );
      rc->conv_pipe->dump_ios( *out, rc->conv_ios );
    }

    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      imgs->load_all_imgs();
      run_cnet->setup_cnet(); 
      for( vect_p_img_t::const_iterator i = imgs->all_imgs->begin(); i != imgs->all_imgs->end(); ++i ) {
	run_cnet_dense->in_sz.max_eq( (*i)->sz );
      }
      run_cnet_dense->setup_cnet();

      in_img = make_p_img_t( run_cnet->in_sz );
      in_img_dense = make_p_img_t( run_cnet_dense->in_sz );

      dump_pipe_and_ios( run_cnet );
      dump_pipe_and_ios( run_cnet_dense );

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

	// figure out what part of output (if any) doesn't depend on padding
	i32_box_t feat_box;
	in_box_to_out_box( feat_box, u32_box_t(u32_pt_t{},run_cnet->in_sz), cm_valid, *run_cnet->ol_csi );

	for( uint32_t wix = 0; wix != wins_per_image; ++wix ) {
	  u32_pt_t const samp_nc_max = (*i)->sz - run_cnet->in_sz;
	  u32_pt_t const samp_nc = random_pt( samp_nc_max, gen );
	  ++tot_wins;
	  comp_win( feat_box, out_batch_dense, (*i), samp_nc );
	}
      }
      out.reset();
    }
    void comp_win( i32_box_t const & feat_box, p_nda_float_t out_batch_dense, p_img_t const & img, u32_pt_t const & nc ) {
      dims_t const & obd_dense = out_batch_dense->dims;
      assert( obd_dense.sz() == 4 );
      assert( obd_dense.dims(0) == 1 ); // one image
      //printf( "nc=%s\n", str(nc).c_str() );   
      u32_box_t in_box{ nc, nc + run_cnet->in_sz };

      i32_box_t feat_box_dense;
      in_box_to_out_box( feat_box_dense, in_box, cm_valid, *run_cnet_dense->ol_csi );

      if( feat_box_dense.sz() != feat_box.sz() ) { return; }
      
      (*out) << strprintf( "feat_box_dense=%s\n", str(feat_box_dense).c_str() );
      // run net on just sample area
      img_copy_to_clip( img.get(), in_img.get(), {}, nc );
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, in_img );
      p_nda_float_t out_batch;
      {
	timer_t t1("sparse_cnn");
	out_batch = run_cnet->run_one_blob_in_one_blob_out();
	p_nda_float_t feats = feats_copy_clip( out_batch, feat_box );
	p_nda_float_t feats_dense = feats_copy_clip( out_batch_dense, feat_box_dense );
	(*out) << strprintf( "ssds_str(from_dense,out_batch)=%s\n", str(ssds_str(feats_dense,feats)).c_str() );
      }
    }
  };

  struct test_upsamp_t : virtual public nesi, public has_main_t // NESI( help="test img vs. filt upsamp",
			// bases=["has_main_t"], type_id="test_upsamp")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test_upsamp.txt",help="output: text summary of differences between net and img based-upsampling features computation.")
    p_load_imgs_from_pascal_classes_t imgs;//NESI(default="()")
    string model_name; //NESI(default="nin_imagenet_nopad",help="name of model")
    p_run_cnet_t run_cnet; //NESI(default="(in_sz=516 516,ptt_fn=%(models_dir)/%(model_name)/deploy.prototxt.boda
                           // ,trained_fn=%(models_dir)/%(model_name)/best.caffemodel
                           // ,out_layer_name=relu12)",help="CNN model params")
    p_run_cnet_t run_cnet_upsamp; //NESI(default="(in_sz=258 258,ptt_fn=%(models_dir)/%(model_name)/deploy-in-2X-us.prototxt.boda" 
                                 // ",trained_fn=%(models_dir)/%(model_name)/best.caffemodel"
                                 // ",out_layer_name=relu12)",help="CNN model params")
    uint32_t wins_per_image; //NESI(default="1",help="number of random windows per image to test")

    string upsamp_layer_name; //NESI(default="conv1",help="name of layer to downsample filters of into upsamp net")
    
    p_img_t in_img;
    p_img_t in_img_upsamp;

    p_conv_pipe_t conv_pipe;
    p_vect_conv_io_t conv_ios;

    p_conv_pipe_t conv_pipe_upsamp;
    p_vect_conv_io_t conv_ios_upsamp;

    void get_pipe_and_ios( p_run_cnet_t const & rc, p_conv_pipe_t & conv_pipe, p_vect_conv_io_t & conv_ios ) {
      conv_pipe = rc->conv_pipe;
      //conv_pipe->dump_pipe( *out );
      conv_ios = conv_pipe->calc_sizes_forward( rc->in_sz, 0 ); 
      //conv_pipe->dump_ios( *out, conv_ios );
    }
    
    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      imgs->load_all_imgs();
      run_cnet->setup_cnet(); 
      run_cnet_upsamp->setup_cnet();

      // FIXME/TODO/TESTING: dump dims of layer blobs ...
      vect_p_nda_float_t usl_blobs;
      copy_layer_blobs( run_cnet->net, upsamp_layer_name, usl_blobs );

      vect_p_nda_float_t usl_blobs_upsamp;
      copy_layer_blobs( run_cnet_upsamp->net, upsamp_layer_name + "-in-2X-us", usl_blobs_upsamp );

      assert_st( usl_blobs.size() == 2 ); // filters, biases
      assert_st( usl_blobs_upsamp.size() == 2 ); // filters, biases
      assert_st( usl_blobs[1]->dims == usl_blobs_upsamp[1]->dims ); // biases should be same shape (and same strides?)
      usl_blobs_upsamp[1] = usl_blobs[1]; // use biases unchanged in upsamp net
      assert_st( usl_blobs[0]->dims.dims(0) == usl_blobs_upsamp[0]->dims.dims(0) );
      assert_st( usl_blobs[0]->dims.dims(1) == usl_blobs_upsamp[0]->dims.dims(1) );
      assert_st( u32_ceil_div( usl_blobs[0]->dims.dims(2), 2 ) == usl_blobs_upsamp[0]->dims.dims(2) );
      assert_st( u32_ceil_div( usl_blobs[0]->dims.dims(3), 2 ) == usl_blobs_upsamp[0]->dims.dims(3) );

      for( dims_iter_t di( usl_blobs_upsamp[0]->dims ) ; ; ) { usl_blobs_upsamp[0]->at(di.di) = 0; 
	if( !di.next() ) { break; } 
      }

      for( dims_iter_t di( usl_blobs[0]->dims ) ; ; ) { 
	usl_blobs_upsamp[0]->at4(di.di[0],di.di[1],di.di[2]>>1,di.di[3]>>1) += usl_blobs[0]->at( di.di );
	if( !di.next() ) { break; } 
      }

      set_layer_blobs( run_cnet_upsamp->net, upsamp_layer_name + "-in-2X-us", usl_blobs_upsamp );
      

      // in_img = make_p_img_t( run_cnet->in_sz ); // re-created each use by upsampling
      in_img_upsamp = make_p_img_t( run_cnet_upsamp->in_sz );

      get_pipe_and_ios( run_cnet, conv_pipe, conv_ios );
      get_pipe_and_ios( run_cnet_upsamp, conv_pipe_upsamp, conv_ios_upsamp );
      
      boost::random::mt19937 gen;

      uint32_t tot_wins = 0;
      for( vect_p_img_t::const_iterator i = imgs->all_imgs->begin(); i != imgs->all_imgs->end(); ++i ) {
	u32_pt_t const samp_sz = run_cnet_upsamp->in_sz;
	if( !(*i)->sz.both_dims_ge( samp_sz ) ) { continue; } // img too small to sample. assert? warn?
	(*out) << strprintf( "(*i)->sz=%s\n", str((*i)->sz).c_str() );

	for( uint32_t wix = 0; wix != wins_per_image; ++wix ) {
	  u32_pt_t const samp_nc_max = (*i)->sz - run_cnet_upsamp->in_sz;
	  u32_pt_t const samp_nc = random_pt( samp_nc_max, gen );
	  ++tot_wins;
	  comp_win( (*i), samp_nc );
	}
      }
      out.reset();
    }

    void comp_win( p_img_t const & img, u32_pt_t const & nc ) {
      // run both net on entire input image. first, use the net with built-in 2x upsampling
      img_copy_to_clip( img.get(), in_img_upsamp.get(), {}, nc );
      subtract_mean_and_copy_img_to_batch( run_cnet_upsamp->in_batch, 0, in_img_upsamp );
      p_nda_float_t out_batch_upsamp;
      {
	timer_t t1("net_upsamp_cnn");
	out_batch_upsamp = run_cnet_upsamp->run_one_blob_in_one_blob_out();
      }
      // next, upsample image an run using normal/stock net
      in_img = upsample_2x( in_img_upsamp );
      assert_st( in_img->sz == run_cnet->in_sz );
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, in_img );
      p_nda_float_t out_batch;
      {
	timer_t t1("img_upsamp_cnn");
	out_batch = run_cnet->run_one_blob_in_one_blob_out();
      }
      (*out) << strprintf( "ssds_str(out_batch_upsamp,out_batch)=%s\n", 
			   str(ssds_str(out_batch_upsamp,out_batch)).c_str() );
    }
  };

#include"gen/test_dense.cc.nesi_gen.cc"
  
}
