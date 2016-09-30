// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"build_info.H"
#include"str_util.H"
#include"has_main.H"
#include"comp_util.H"
#include"rand_util.H"
#include"results_io.H"
#include"caffeif.H"
#include"img_io.H"
#include"imagenet_util.H"
#include"nesi.H" // for str(nesi)

namespace boda 
{


  // example test_compute command line for testing nvrtc:
  // time boda test_compute --model-name=nin_imagenet_nopad --wins-per-image=1 --imgs='(pil_fn=%(boda_test_dir)/pascal/head_1/%%s.txt)' --run-cnet='(in_dims=(img=256),ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=pool4)' --use-nvrtc=1 --max-err=10 && cat test_compute.txt

  // matching cnet_ana+flops.py command line:
  // boda cnet_ana --in-model=nin_imagenet_nopad --print-ops=1 --in-dims='(img=1)' && python ../../pysrc/flops.py --per-layer=1 --ai-mnk=1 --per-layer-in-info=1

  struct test_compute_t : virtual public nesi, public has_main_t // NESI( help="comparison test 2 CNN computation methods",
			// bases=["has_main_t"], type_id="test_compute")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test_compute.txt",help="output: text summary of differences between computations.")
    p_load_pil_t imgs;//NESI(default="()")
    p_run_cnet_t run_cnet; //NESI(help="CNN model params")
    uint32_t wins_per_image; //NESI(default="10",help="number of random windows per image to test")

    p_has_conv_fwd_t cf1; //NESI(default="(mode=caffe)",help="fwd compute mode 1; used if cm1==1")
    p_has_conv_fwd_t cf2; //NESI(help="fwd compute mode 2; used if cm2==1")

    uint32_t tpd; //NESI(default="0",help="if non-zero, use test-pattern data. 1 == const, 2 == const + x co-ord")
    u32_pt_t tpd_in_sz; //NESI(default="15 15",help="x,y size of test-pattern data to use")
    double tpd_const; //NESI(default="1.0",help="test-pattern data constant offset")

    uint32_t diff_show_mrd_only; //NESI(default="0",help="if 1, print only MAD for diffs, not full sds_diff_t. usefull for making test outputs for 'pseudo-failure' consistent (such as quantization tests where specific numerical errors are expected.")
    double mrd_toler; //NESI(default="5e-4",help="maximum maximum-absolute-difference over which a failure is declared")
    map_str_double var_mrd_toler; //NESI(default="()",help="per-layer custom maximum maximum-absolute-differences over which a failure is declared (overrides mrd_toler per-layer if specified")

    uint32_t max_err; //NESI(default="10",help="print at most this many differing elems")

    p_img_t in_img;

    uint32_t num_mad_fail;

    vect_string tops; //NESI(help="vars to check")

    uint32_t show_digests; //NESI(default="0",help="if non-zero, show nda digests for cf1" )
    uint32_t cmp_digests; //NESI(default="0",help="if non-zero, compare nda digests for cf1 and cf2" )
    uint32_t write_digests; //NESI(default="0",help="if non-zero, write nda digests for cf1 and cf2" )
    filename_t digest_fn; //NESI(default="%(boda_output_dir)/digest-%%s.boda",help="output: binary stream of digests of all ndas. %%s will be replaced with the engine backend index (i.e. '1' or '2')")
    double cf1_digest_self_mrd_toler; //NESI(default="0.0",help="maximum maximum-absolute-difference over which a failure is declared (cf1 digest self-cmp case only)")

    void dump_pipe_and_ios( p_run_cnet_t const & rc ) {
      rc->conv_pipe->dump_pipe( *out );
      rc->conv_pipe->dump_ios( *out );
      rc->conv_pipe->dump_ops( *out );
    }

    p_ostream out;

    p_ostream digest_out1;
    p_ostream digest_out2;

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

#if 0      
      // FIXME/HACK: for now, set tolerance for caffe + bck runs (presuming usage of non-deterministic cuDNN)
      if( (cf1->mode == "caffe") && (run_cnet->add_bck_ops == 1) ) { cf1_digest_self_mrd_toler = 5e-5; }

      printf( "\n*this=%s\n", str(*this).c_str() );
      printf( "run_cnet=%s\n", str(run_cnet).c_str() );
      if( cf1 ) { printf( "cf1=%s\n", str(*cf1).c_str() ); }
      if( cf2 ) { printf( "cf2=%s\n", str(*cf2).c_str() ); }
#endif

      if( write_digests ) {
        if( cf1 ) { digest_out1 = ofs_open( strprintf( digest_fn.exp.c_str(), str(uint32_t(1)).c_str() ) ); 
          bwrite( *digest_out1, boda_magic ); }
        if( cf2 ) { digest_out2 = ofs_open( strprintf( digest_fn.exp.c_str(), str(uint32_t(2)).c_str() ) );
          bwrite( *digest_out2, boda_magic ); }
      }
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
	    // FIXME: make seed be tot_wins * size_of_image * images_per_batch or the like?
	    if( cf1 ) { cf1->set_det_drop_seed( tot_wins ); }
	    if( cf2 ) { cf2->set_det_drop_seed( tot_wins ); }
	    comp_batch();
	  }
	}
      }
      // FIXME: delimit the two info logs somehow? but if they are empty, we don't want to output clutter ...
      if( cf1 ) { (*out) << cf1->get_info_log(); }
      if( cf2 ) { (*out) << cf2->get_info_log(); }
      if( !num_mad_fail ) { (*out) << strprintf( "***ALL IS WELL***\n" ); }
      else { (*out) << strprintf( "***MAD FAILS*** num_mad_fail=%s\n", str(num_mad_fail).c_str() ); }
      out.reset();
      if( write_digests ) {
        if( cf1 ) { bwrite( *digest_out1, string("END_BODA") ); }
        if( cf2 ) { bwrite( *digest_out2, string("END_BODA") ); }
      }
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
      p_map_str_p_nda_float_t fwd1 = make_shared<map_str_p_nda_float_t>();
      p_map_str_p_nda_float_t fwd2 = make_shared<map_str_p_nda_float_t>();
      vect_string to_set_vns1;
      vect_string to_set_vns2;
      run_cnet->conv_pipe->run_setup_input( run_cnet->in_batch, fwd1, to_set_vns1 );
      run_cnet->conv_pipe->run_setup_input( run_cnet->in_batch, fwd2, to_set_vns2 );
      if( tops.empty() ) {
	string const & onn = run_cnet->conv_pipe->out_node_name;
	if( !onn.empty() ) { tops.push_back( onn ); } 
	else { run_cnet->conv_pipe->get_topo_order_caffe_comp_nodes( tops ); }
      }
      if( cf1 ) { cf1->run_fwd( to_set_vns1, fwd1, tops ); }
      if( cf2 ) { cf2->run_fwd( to_set_vns2, fwd2, tops ); }
      bool const do_cmp = bool(cf1) && bool(cf2);
      if( do_cmp ) { 
	comp_vars( out.get(), num_mad_fail,
		   mrd_toler, &var_mrd_toler,
		   diff_show_mrd_only, max_err, 
		   tops, fwd1, fwd2 );
      }
      for( vect_string::const_iterator i = tops.begin(); i != tops.end(); ++i ) {
        size_t const digest_seed = std::hash<string>()(*i);
        p_nda_digest_t digest1;
        p_nda_digest_t digest2;
        if( (do_cmp && cmp_digests) || write_digests || show_digests ) {
          if( cf1 ) { digest1 = nda_digest_t::make_from_nda( must_find( *fwd1, *i ), digest_seed ); digest1->self_cmp_mrd = cf1_digest_self_mrd_toler;}
          if( cf2 ) { digest2 = nda_digest_t::make_from_nda( must_find( *fwd2, *i ), digest_seed ); }
        }
        if( show_digests ) {
          if( cf1 ) { printf( "cf1 %s digest_str=%s\n", str((*i)).c_str(), digest1->get_digest().c_str() ); }
          if( cf2 ) { printf( "cf2 %s digest_str=%s\n", str((*i)).c_str(), digest2->get_digest().c_str() ); }
        }
        if( write_digests ) {
          if( cf1 ) { bwrite_id( *digest_out1, *i ); bwrite( *digest_out1, string("p_nda_digest_t") ); bwrite( *digest_out1, digest1 ); }
          if( cf2 ) { bwrite_id( *digest_out2, *i ); bwrite( *digest_out2, string("p_nda_digest_t") ); bwrite( *digest_out2, digest2 ); }
        }
        if( do_cmp && cmp_digests ) {
          double vmt = get( var_mrd_toler, *i, mrd_toler );
          string const comp_res = digest1->mrd_comp( digest2, vmt );
          if( !comp_res.empty() ) { (*out) << (*i) + " digest mrd_comp() failure:\n" + comp_res + "\n"; } 
        }
      }
    }
  };


  struct gen_test_compute_tests_t : virtual public nesi, public has_main_t // NESI( help="generate list of test_compute tests",
			// bases=["has_main_t"], type_id="gen_test_compute_tests")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    virtual void main( nesi_init_arg_t * nia ) {
      string input_cfg = "--wins-per-image=1 --imgs='(pil_fn=%(boda_test_dir)/pascal/head_1/%%s.txt)'";
      vect_pair_str_str model_cfgs = {
        {"alexnet","--model-name=alexnet_ng_conv --run-cnet='(in_dims=(img=20))'"},
        // note: mutated version just moves a ReLU after a norm, so there is a non-fused relu
        {"alexnet_mutated","--model-name=alexnet_ng_conv_mutated_for_testing --run-cnet='(in_dims=(img=20))'"}, 
        {"nin","--model-name=nin_imagenet --run-cnet='(in_dims=(img=20,y=227,x=227))'"},
        {"goog","--model-name=googlenet_conv --run-cnet='(in_dims=(img=20,y=227,x=227))'"},
        // grad tests
        {"grad_goog","--model-name=googlenet_conv --run-cnet='(in_dims=(img=5,y=227,x=227),add_bck_ops=1)'"},
        {"grad_nin","--model-name=nin_imagenet --run-cnet='(in_dims=(img=1),add_bck_ops=1)'"},
        {"grad_alenet_nd_nl","--model-name=alexnet_ng_conv_nd_nl --run-cnet='(in_dims=(img=1),add_bck_ops=1)'"},
        {"grad_alexnet","--model-name=alexnet_ng_conv --run-cnet='(in_dims=(img=1),add_bck_ops=1)'"},
        {"grad_firenet","--model-name=firenet8-CE-0.125-nofinalpad --run-cnet='(in_dims=(img=3))'"},
        {"grad_bconv_strides","--model-name=bconv_strides_1 --run-cnet='(in_dims=(img=1),add_bck_ops=1)'"}
      };
      
      vect_pair_str_str run_cfgs;
      if( is_feature_enabled("caffe") ) { run_cfgs.push_back( {"caffe","mode=caffe"} ); }
      if( is_feature_enabled("nvrtc") ) { run_cfgs.push_back( {"rtc_nvrtc","mode=rtc,rtc=(be=nvrtc)"} ); }
      if( is_feature_enabled("opencl") ) { run_cfgs.push_back( {"rtc_ocl","mode=rtc,rtc=(be=ocl)"} ); }

      string const rtc_opt = ",op_tune=(k1conv=1,tconv=1),enable_write_xpose=1"; // for rtc mode; orig. tests only for rtc+nvrtc submode
      string const rtc_bconv = ",enable_bconv=1"; // for rtc mode, enables optimized bck ops
      
      // for now, we only handle one input cfg, but that would be easy to change
      string const test_cli = "test_compute " + input_cfg;
      for( vect_pair_str_str::const_iterator i = model_cfgs.begin(); i != model_cfgs.end(); ++i ) {
        string const tn_i = i->first;
        string const tc_i = test_cli + " " + i->second;
        vect_string cfs;
        for( vect_pair_str_str::iterator j = run_cfgs.begin(); j != run_cfgs.end(); ++j ) {
          string const tn_j = tn_i + "_" + j->first;
          if( !startswith( i->first, "grad_" ) ) { // for non-grad (i.e. regular fwd) tests, do reg and opt (rtc only) tests
            cfs.push_back( j->second );
            if( startswith( j->first, "rtc_" ) ) { cfs.push_back( j->second + rtc_opt ); }
          } else { // for grad tests, do opt and opt+bconv tests for rtc, just reg for caffe
            if( startswith( j->first, "rtc_" ) ) { 
              cfs.push_back( j->second + rtc_opt ); 
              cfs.push_back( j->second + rtc_bconv ); 
            } else {
              cfs.push_back( j->second ); 
            }
          }
        }
        printf( "tn_i=%s cfs=%s\n", str(tn_i).c_str(), str(cfs).c_str() );
      }
// "--cf2=(mode=rtc,quantize=(li_0=(name=conv1,max_val=1024,keep_bits=9)),op_tune=(tconv_max_ksz=11 11))" // quantize
// "--cf2=(mode=rtc,enable_stats=1,op_tune=(tconv_max_ksz=11 11))" // stats

                                
    }
  };

#include"gen/test_compute.cc.nesi_gen.cc"
  
}
