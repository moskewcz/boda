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

namespace boda 
{

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
    b.tn = e.tn;
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
    double mrd_toler; //NESI(default="5e-5",help="maximum maximum-absolute-difference over which a failure is declared")

    p_img_t in_img;
    p_img_t in_img_dense;
    uint32_t num_mad_fail;

    void dump_pipe_and_ios( p_run_cnet_t const & rc ) {
      rc->conv_pipe->dump_pipe( *out );
      rc->conv_pipe->dump_ios( *out );
    }

    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      imgs->load_img_db( 1 );
      run_cnet->setup_cnet( nia );
      u32_pt_t const max_img_sz = imgs->img_db->get_max_img_sz();
      run_cnet_dense->in_dims["y"] = max_img_sz.d[1]; run_cnet_dense->in_dims["x"] = max_img_sz.d[0];
      run_cnet_dense->setup_cnet( nia );

      in_img = make_p_img_t( run_cnet->conv_pipe->get_data_img_xy_dims_3_chans_only() );
      in_img_dense = make_p_img_t( run_cnet_dense->conv_pipe->get_data_img_xy_dims_3_chans_only() );

      dump_pipe_and_ios( run_cnet );
      dump_pipe_and_ios( run_cnet_dense );

      boost::random::mt19937 gen;
      num_mad_fail = 0;
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
      if( !num_mad_fail ) { (*out) << strprintf( "***ALL IS WELL***\n" ); }
      else { (*out) << strprintf( "***MAD FAILS*** num_mad_fail=%s\n", str(num_mad_fail).c_str() ); }
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
	ssds_diff_t const ssds_diff(feats_dense,feats);
	bool is_fail = 0;
	if( (ssds_diff.mrd >= mrd_toler) || ssds_diff.has_nan() ) { ++num_mad_fail; is_fail = 1; }
	if( is_fail ) { (*out) << strprintf( "ssds_diff_t(feats_dense,feats)=%s\n", str(ssds_diff).c_str() ); }
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
    double mrd_toler; //NESI(default="2e-4",help="maximum maximum-absolute-difference over which a failure is declared")

    p_img_t in_img;
    p_img_t in_img_upsamp;

    uint32_t num_mad_fail;
    
    p_ostream out;
    virtual void main( nesi_init_arg_t * nia ) {
      out = ofs_open( out_fn.exp );
      //out = p_ostream( &std::cout, null_deleter<std::ostream>() );
      imgs->load_img_db( 1 );
      run_cnet->setup_cnet( nia ); 

      boost::random::mt19937 gen;

      uint32_t tot_wins = 0;
      in_img_upsamp = make_p_img_t( run_cnet->conv_pipe->get_data_img_xy_dims_3_chans_only() );
      in_img_upsamp->fill_with_pel( u32_rgba_inmc );
      u32_pt_t const samp_sz = in_img_upsamp->sz >> 1;
      // in_img = make_p_img_t( run_cnet->in_sz ); // re-created each use by upsampling
      num_mad_fail = 0;
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
      if( !num_mad_fail ) { (*out) << strprintf( "***ALL IS WELL***\n" ); }
      else { (*out) << strprintf( "***MAD FAILS*** num_mad_fail=%s\n", str(num_mad_fail).c_str() ); }

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

      ssds_diff_t const ssds_diff(feats_upsamp,feats);
      bool is_fail = 0;
      if( (ssds_diff.mrd >= mrd_toler) || ssds_diff.has_nan() ) { ++num_mad_fail; is_fail = 1; }
      if( is_fail ) { (*out) << strprintf( "ssds_diff_t(feats_upsamp,feats)=%s\n", str(ssds_diff).c_str() ); }
    }
  };


  
#include"gen/test_dense.cc.nesi_gen.cc"
  
}
