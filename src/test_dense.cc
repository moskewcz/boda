// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"caffeif.H"
#include"caffepb.H"
#include"conv_util.H"
#include"rand_util.H"
#include"timers.H"
#include"imagenet_util.H"

namespace boda {

  template< typename T >
  shared_ptr< nda_T< T > > copy_clip( shared_ptr< nda_T< T > > const & in, dims_t const & b, dims_t const & e ) { 
    assert_st( b.fits_in( in->dims ) );
    assert_st( e.fits_in( in->dims ) );
    dims_t ret_dims = e - b;
    shared_ptr< nda_T< T > > ret( new nda_T< T >( ret_dims ) );
    dims_iter_t rdi( ret_dims );
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
    p_load_pil_t imgs;//NESI(default="()")
    p_run_cnet_t run_cnet; //NESI(default="(in_dims=(img=1),out_node_name=cccp8)",help="CNN model params")
    p_run_cnet_t run_cnet_dense; //NESI(default="(in_dims=(img=1),out_node_name=cccp8)",help="CNN model params")
    uint32_t wins_per_image; //NESI(default="1",help="number of random windows per image to test")

    p_img_t in_img;
    p_img_t in_img_dense;

    void dump_pipe_and_ios( p_run_cnet_t const & rc ) {
      rc->conv_pipe->dump_pipe( *out );
      rc->conv_pipe->dump_ios( *out );
    }

    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      imgs->load_img_db( 1 );
      run_cnet->setup_cnet();
      u32_pt_t const max_img_sz = imgs->img_db->get_max_img_sz();
      run_cnet_dense->in_dims["y"] = max_img_sz.d[1]; run_cnet_dense->in_dims["x"] = max_img_sz.d[0];
      run_cnet_dense->setup_cnet();

      in_img = make_p_img_t( run_cnet->conv_pipe->get_data_img_xy_dims_3_chans_only() );
      in_img_dense = make_p_img_t( run_cnet_dense->conv_pipe->get_data_img_xy_dims_3_chans_only() );

      dump_pipe_and_ios( run_cnet );
      dump_pipe_and_ios( run_cnet_dense );

      boost::random::mt19937 gen;

      uint32_t tot_wins = 0;
      for( vect_p_img_info_t::const_iterator i = imgs->img_db->img_infos.begin(); i != imgs->img_db->img_infos.end(); ++i ) {
	if( !(*i)->img->sz.both_dims_ge( in_img->sz ) ) { continue; } // img too small to sample. assert? warn?
	(*out) << strprintf( "(*i)->sz=%s\n", str((*i)->img->sz).c_str() );
	// run net on entire input image
	in_img_dense->fill_with_pel( u32_rgba_inmc );
	img_copy_to_clip( (*i)->img.get(), in_img_dense.get() );
	subtract_mean_and_copy_img_to_batch( run_cnet_dense->in_batch, 0, in_img_dense );
	p_nda_float_t out_batch_dense;
	{
	  timer_t t1("dense_cnn");
	  out_batch_dense = run_cnet_dense->run_one_blob_in_one_blob_out();
	}

	// figure out what part of output (if any) doesn't depend on padding
	i32_box_t feat_box;
	in_box_to_out_box( feat_box, u32_box_t(u32_pt_t{},in_img->sz), cm_valid, run_cnet->get_out_csi(0) );

	for( uint32_t wix = 0; wix != wins_per_image; ++wix ) {
	  u32_pt_t const samp_nc_max = (*i)->img->sz - in_img->sz;
	  u32_pt_t const samp_nc = random_pt( samp_nc_max, gen );
	  ++tot_wins;
	  comp_win( feat_box, out_batch_dense, (*i)->img, samp_nc );
	}
      }
      out.reset();
    }
    void comp_win( i32_box_t const & feat_box, p_nda_float_t out_batch_dense, p_img_t const & img, u32_pt_t const & nc ) {
      dims_t const & obd_dense = out_batch_dense->dims;
      assert( obd_dense.sz() == 4 );
      assert( obd_dense.dims(0) == 1 ); // one image
      //printf( "nc=%s\n", str(nc).c_str() );   
      u32_box_t in_box{ nc, nc + in_img->sz };

      i32_box_t feat_box_dense;
      in_box_to_out_box( feat_box_dense, in_box, cm_valid, run_cnet_dense->get_out_csi(0) );

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
	//float const sum_f = nda_reduce( *feats, sum_functor<float>(), 0.0f );
	//float const sum_fd = nda_reduce( *feats_dense, sum_functor<float>(), 0.0f );
	//(*out) << strprintf( "sum_f=%s sum_fd=%s\n", str(sum_f).c_str(), str(sum_fd).c_str() );
	(*out) << strprintf( "ssds_str(from_dense,out_batch)=%s\n", str(ssds_diff_t(feats_dense,feats)).c_str() );
      }
    }
  };

  struct test_upsamp_t : virtual public nesi, public has_main_t // NESI( help="test img vs. filt upsamp",
			// bases=["has_main_t"], type_id="test_upsamp")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test_upsamp.txt",help="output: text summary of differences between net and img based-upsampling features computation.")
    p_load_pil_t imgs;//NESI(default="()")
    p_run_cnet_t run_cnet; //NESI(default="(enable_upsamp_net=1,in_dims=(img=1,y=516,x=516),out_node_name=cccp8)",help="CNN model params")
    uint32_t wins_per_image; //NESI(default="1",help="number of random windows per image to test")

    string upsamp_layer_name; //NESI(default="conv1",help="name of layer to downsample filters of into upsamp net")
    
    p_img_t in_img;
    p_img_t in_img_upsamp;
    
    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      imgs->load_img_db( 1 );
      run_cnet->setup_cnet(); 

      boost::random::mt19937 gen;

      uint32_t tot_wins = 0;
      in_img_upsamp = make_p_img_t( run_cnet->conv_pipe->get_data_img_xy_dims_3_chans_only() );
      in_img_upsamp->fill_with_pel( u32_rgba_inmc );
      u32_pt_t const samp_sz = in_img_upsamp->sz >> 1;
      // in_img = make_p_img_t( run_cnet->in_sz ); // re-created each use by upsampling
      for( vect_p_img_info_t::const_iterator i = imgs->img_db->img_infos.begin(); i != imgs->img_db->img_infos.end(); ++i ) {
	if( !(*i)->img->sz.both_dims_ge( samp_sz ) ) { continue; } // img too small to sample. assert? warn?
	(*out) << strprintf( "(*i)->sz=%s\n", str((*i)->img->sz).c_str() );
	for( uint32_t wix = 0; wix != wins_per_image; ++wix ) {
	  u32_pt_t const samp_nc_max = (*i)->img->sz - samp_sz;
	  u32_pt_t const samp_nc = random_pt( samp_nc_max, gen );
	  ++tot_wins;
	  comp_win( samp_sz, (*i)->img, samp_nc );
	}
      }
      out.reset();
    }

    void comp_win( u32_pt_t const & samp_sz, p_img_t const & img, u32_pt_t const & nc ) {
      // run both net on entire input image. first, use the net with
      // built-in 2x upsampling note that 3/4 of the input image is
      // empty here. this is not really needed/sensible, but it allows
      // this testing usage of upsampling to match the 'normal' usage
      // where the same input image is used for the regular and
      // upsampled nets.
      img_copy_to_clip( img.get(), in_img_upsamp.get(), {}, nc, samp_sz );
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, in_img_upsamp );
      p_nda_float_t out_batch_upsamp;
      {
	timer_t t1("net_upsamp_cnn");
	out_batch_upsamp = run_cnet->run_one_blob_in_one_blob_out_upsamp();
      }
      // next, upsample image an run using normal/stock net
      in_img = make_p_img_t( samp_sz );
      img_copy_to_clip( img.get(), in_img.get(), {}, nc );
      in_img = upsample_2x( in_img );
      assert_st( in_img->sz == in_img_upsamp->sz );
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, in_img );
      p_nda_float_t out_batch;
      {
	timer_t t1("img_upsamp_cnn");
	out_batch = run_cnet->run_one_blob_in_one_blob_out();
      }
      //printf( "out_batch->dims=%s out_batch_upsamp->dims=%s\n", str(out_batch->dims).c_str(), str(out_batch_upsamp->dims).c_str() );
      // bear in mind that nets which use padding may complicate this comparison
      u32_box_t in_box{ {}, samp_sz };
      i32_box_t feat_box_upsamp;
      in_box_to_out_box( feat_box_upsamp, in_box, cm_valid, run_cnet->get_out_csi(1) );

      i32_box_t feat_box;
      in_box_to_out_box( feat_box, u32_box_t{{},in_img_upsamp->sz}, cm_valid, run_cnet->get_out_csi(0) );
      
      //printf( "feat_box=%s feat_box_upsamp=%s\n", str(feat_box).c_str(), str(feat_box_upsamp).c_str() );
      assert_st( feat_box_upsamp.sz() == feat_box.sz() );

      p_nda_float_t feats_upsamp = feats_copy_clip( out_batch_upsamp, feat_box_upsamp );
      // note: if there is no padding, there is be no need to clip, use we could just all features
      // p_nda_float_t feats = out_batch; 
      p_nda_float_t feats = feats_copy_clip( out_batch, feat_box );

      (*out) << strprintf( "ssds_str(out_batch_upsamp,out_batch)=%s\n", 
			   str(ssds_diff_t(feats_upsamp,feats)).c_str() );
    }
  };


  // example test_compute command line for testing nvrtc:
  // time boda test_compute --model-name=nin_imagenet_nopad --wins-per-image=1 --imgs='(pil_fn=%(boda_test_dir)/pascal/head_1/%%s.txt)' --run-cnet='(in_dims=(img=256),ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=pool4)' --use-nvrtc=1 --max-err=10 && cat test_compute.txt

  // matching cnet_ana+flops.py command line:
  // boda cnet_ana --in-model=nin_imagenet_nopad --print-ops=1 --in-sz=227 --out-layer-name=relu0  && python ../../pysrc/flops.py --per-layer=1 --backward=0 --runtime=.015 --num-imgs=20

  struct test_compute_t : virtual public nesi, public has_main_t // NESI( help="comparison test 2 CNN computation methods",
			// bases=["has_main_t"], type_id="test_compute")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test_compute.txt",help="output: text summary of differences between computations.")
    p_load_pil_t imgs;//NESI(default="()")
    p_run_cnet_t run_cnet; //NESI(help="CNN model params")
    uint32_t wins_per_image; //NESI(default="10",help="number of random windows per image to test")

    p_has_conv_fwd_t cf1; //NESI(default="(mode=caffe)",help="fwd compute mode 1; used if cm1==1")
    p_has_conv_fwd_t cf2; //NESI(default="(mode=caffe)",help="fwd compute mode 2; used if cm2==1")

    uint32_t tpd; //NESI(default="0",help="if non-zero, use test-pattern data. 1 == const, 2 == const + x co-ord")
    u32_pt_t tpd_in_sz; //NESI(default="15 15",help="x,y size of test-pattern data to use")
    double tpd_const; //NESI(default="1.0",help="test-pattern data constant offset")

    double mad_toler; //NESI(default="1e-5",help="maximum maximum-absolute-difference over which a failure is declared")

    uint32_t max_err; //NESI(default="10",help="print at most this many differing elems")

    p_img_t in_img;

    uint32_t num_mad_fail;

    void dump_pipe_and_ios( p_run_cnet_t const & rc ) {
      rc->conv_pipe->dump_pipe( *out );
      rc->conv_pipe->dump_ios( *out );
      rc->conv_pipe->dump_ops( *out );
    }

    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      if( tpd ) { run_cnet->in_dims["y"] = tpd_in_sz.d[1]; run_cnet->in_dims["x"] = tpd_in_sz.d[0]; }
      run_cnet->setup_cnet(); 

      if( !tpd ) { 
	imgs->load_img_db( 1 ); 
	in_img = make_p_img_t( run_cnet->conv_pipe->get_data_img_xy_dims_3_chans_only() ); 
	in_img->fill_with_pel( u32_rgba_inmc ); 
      }

      if( cf1 ) { cf1->init( run_cnet->conv_pipe ); }
      if( cf2 ) { cf2->init( run_cnet->conv_pipe ); }

      //dump_pipe_and_ios( run_cnet );
      
      boost::random::mt19937 gen;
      num_mad_fail = 0;
      uint32_t tot_wins = 0;
      if( tpd ) {
	make_tpd_batch( run_cnet->in_batch );
	comp_batch();
      } else {
	for( vect_p_img_info_t::const_iterator i = imgs->img_db->img_infos.begin(); i != imgs->img_db->img_infos.end(); ++i ) {
	  (*out) << strprintf( "(*i)->sz=%s\n", str((*i)->img->sz).c_str() );
	  for( uint32_t wix = 0; wix != wins_per_image; ++wix ) {
	    if( !(*i)->img->sz.both_dims_ge( in_img->sz ) ) { 
	      // img too small to sample. use whole image
	      copy_win_to_batch( (*i)->img, u32_pt_t() );
	    } else {
	      u32_pt_t const samp_nc_max = (*i)->img->sz - in_img->sz;
	      u32_pt_t const samp_nc = random_pt( samp_nc_max, gen );
	      copy_win_to_batch( (*i)->img, samp_nc );
	    }
	    ++tot_wins;
	    comp_batch();
	  }
	}
      }
      // FIXME: delimit the two info logs somehow? but if they are empty, we don't want to output clutter ...
      (*out) << cf1->get_info_log();
      (*out) << cf2->get_info_log();
      if( !num_mad_fail ) { (*out) << strprintf( "***ALL IS WELL***\n" ); }
      else { (*out) << strprintf( "***MAD FAILS*** num_mad_fail=%s\n", str(num_mad_fail).c_str() ); }
      out.reset();
    }
    void make_tpd_batch( p_nda_float_t const & in_batch ) {
      dims_t const & ibd = in_batch->dims;
      assert_st( 3 == ibd.dims(1) );
      for( uint32_t img_ix = 0; img_ix != ibd.dims(0); ++img_ix ) {
	for( uint32_t y = 0; y < ibd.dims(2); ++y ) {
	  for( uint32_t x = 0; x < ibd.dims(3); ++x ) {
	    for( uint32_t c = 0; c < 3; ++c ) {
	      float val = tpd_const;
	      if( tpd == 2 ) { val += x; }
	      if( tpd == 3 ) { val += y; }
	      else if( tpd == 4 ) { 
		if( (x==ibd.dims(2)/2) && (y==ibd.dims(3)/2) ) { val += 1.0f; }
	      }
	      // note: RGB -> BGR swap via the '2-c' below
	      in_batch->at4( img_ix, 2-c, y, x ) = val;
	    }
	  }
	}
      }
    }
    void copy_win_to_batch( p_img_t const & img, u32_pt_t const & nc ) {
      // run net on just sample area
      img_copy_to_clip( img.get(), in_img.get(), {}, nc );
      for( uint32_t i = 0; i != run_cnet->in_batch->dims.dims(0); ++i ) {
	subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, i, in_img );
      }
    }
    void comp_batch( void ) {
#if 0
    p_map_str_p_nda_float_t fwd = make_shared<map_str_p_nda_float_t>(); // *op_params );
    vect_string to_set_vns;
    run_setup_input( in, fwd, to_set_vns );
    assert( conv_fwd );
    conv_fwd->run_fwd( to_set_vns, fwd, {get_single_top_node()->name} );
    return must_find( *fwd, get_single_top_node()->name );
#endif
      run_cnet->conv_fwd = cf1;
      p_nda_float_t out_batch_1 = run_cnet->run_one_blob_in_one_blob_out();
      run_cnet->conv_fwd = cf2;
      p_nda_float_t out_batch_2 = run_cnet->run_one_blob_in_one_blob_out();
      // out_batch_2->cm_at1(100) = 45.0; // corrupt a value for sanity checking
      ssds_diff_t const ssds_diff(out_batch_1,out_batch_2);
      bool is_fail = 0;
      if( (ssds_diff.mad >= mad_toler) || ssds_diff.has_nan() ) { ++num_mad_fail; is_fail = 1; }
      vect_uint32_t bad_ixs = { 267093, 270895, 279193 };
      if( is_fail ) { // skip printing errors and details if no mad fail. set mad_toler = 0 to force print (and failure)
	(*out) << strprintf( "ssds_str(out_batch_1,out_batch_2)=%s\n", str(ssds_diff).c_str() );
	uint32_t num_err = 0;
	for( uint32_t i = 0; i != out_batch_1->elems.sz; ++i ) {
	  float const v1 = out_batch_1->cm_at1(i);
	  float const v2 = out_batch_2->cm_at1(i);
	  if( fabs(v1 - v2) >= mad_toler ) {
	    (*out) << strprintf( "i=%s v1=%s v2=%s \n", str(i).c_str(), str(v1).c_str(), str(v2).c_str() );
	    ++num_err;
	    if( num_err > max_err ) { break; }
	  }
	}
      }
    }
  };


#include"gen/test_dense.cc.nesi_gen.cc"
  
}
