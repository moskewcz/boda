// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"conv_util.H"
#include"img_io.H"

#include "caffe/caffe.hpp"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

namespace boda 
{
  using namespace boost;
  using namespace caffe;

  typedef Net< float > Net_float;
  typedef shared_ptr< Net_float > p_Net_float;
  void init_caffe( string const & param_str );
  void raw_do_forward( shared_ptr<Net<float> > net_, vect_p_nda_float_t const & bottom );
  void copy_output_blob_data( shared_ptr<Net<float> > net_, string const & out_layer_name,
				     vect_p_nda_float_t & top );

  p_conv_pipe_t make_p_conv_pipe_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );
  
  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",
		       // bases=["has_main_t"], type_id="conv_pyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t pipe_fn; //NESI(default="%(boda_test_dir)/conv_pyra_pipe.xml",help="input pipe XML filename")
    filename_t ptt_fn; //NESI(default="%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt",help="input net prototxt template filename")
    filename_t trained_fn; //NESI(default="%(home_dir)/alexnet/alexnet_train_iter_470000_v1",help="input trained net from which to copy params")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")
    
    filename_t img_in_fn; //NESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")
    filename_t img_out_fn; //NESI(default="%(boda_output_dir)/out.png",help="output image filename")

    string out_layer_name;//NESI(default="conv5",help="output layer name of which to output top blob of")

    virtual void main( nesi_init_arg_t * nia ) { 

      p_img_t img_in( new img_t );
      img_in->load_fn( img_in_fn.exp );

      p_conv_pipe_t conv_pipe = make_p_conv_pipe_t_init_and_check_unused_from_lexp( parse_lexp_xml_file( pipe_fn.exp ), 0 );

      p_string ptt_str = read_whole_fn( ptt_fn );
      string out_pt_str;
      str_format_from_nvm_str( out_pt_str, *ptt_str, 
			       strprintf( "(xsize=%s,ysize=%s,num=%s,chan=%s)", 
					  str(img_in->w).c_str(), str(img_in->h).c_str(), 
					  str(1).c_str(), str(3).c_str() ) );
      //(*ofs_open( out_fn )) << out_pt_str;
      p_Net_float net = init_caffe( out_pt_str );      
      
      dims_t in_batch_dims( 4 );
      in_batch_dims.dims(3) = img_in->w;
      in_batch_dims.dims(2) = img_in->h;
      in_batch_dims.dims(1) = 3; 
      in_batch_dims.dims(0) = 1; // single-image batch
     
      p_nda_float_t in_batch( new nda_float_t );
      in_batch->set_dims( in_batch_dims );

      uint32_t const inmc = 123U+(117U<<8)+(104U<<16)+(255U<<24); // RGBA
      // copy image to batch
      for( uint32_t y = 0; y < img_in->h; ++y ) {
	for( uint32_t x = 0; x < img_in->w; ++x ) {
	  for( uint32_t c = 0; c < 3; ++c ) {
	    in_batch->at4(0,c,y,x) = img_in->get_pel_chan(x,y,c) - float(uint8_t(inmc >> (c*8)));
	  }
	}
      }

      vect_p_nda_float_t in_data; 
      in_data.push_back( in_batch ); // assume single input blob
      raw_do_forward( net, in_data );

      vect_p_nda_float_t out_data; 
      copy_output_blob_data( net, out_layer_name, out_data );
      
      //printf( "out_data=%s\n", str(out_data).c_str() );
      assert( out_data.size() == 1 ); // assume single output blob
      //p_nda_float_t const & out_batch = in_data.front();
      p_nda_float_t const & out_batch = out_data.front();
      dims_t const & obd = out_batch->dims;
      assert( obd.sz() == 4 );
      assert( obd.dims(0) == 1 ); // should be single-image output batch


      uint32_t sqrt_out_chan = uint32_t( ceil( sqrt( double( obd.dims(1) ) ) ) );
      assert( sqrt_out_chan );
      assert( (sqrt_out_chan*sqrt_out_chan) >= obd.dims(1) );

      p_img_t out_img( new img_t );
      out_img->set_sz_and_alloc_pels( obd.dims(3)*sqrt_out_chan, obd.dims(2)*sqrt_out_chan ); // w, h

      float const out_min = nda_reduce( *out_batch, min_functor<float>(), 0.0f ); // note clamp to 0
      //assert_st( out_min == 0.0f ); // shouldn't be any negative values
      float const out_max = nda_reduce( *out_batch, max_functor<float>(), 0.0f ); // note clamp to 0
      float const out_rng = out_max - out_min;

      

      for( uint32_t y = 0; y < out_img->h; ++y ) {
	for( uint32_t x = 0; x < out_img->w; ++x ) {
	  uint32_t const bx = x / sqrt_out_chan;
	  uint32_t const by = y / sqrt_out_chan;
	  uint32_t const bc = (y%sqrt_out_chan)*sqrt_out_chan + (x%sqrt_out_chan);
	  uint32_t gv;
	  if( bc < obd.dims(1) ) {
	    float const norm_val = ((out_batch->at4(0,bc,by,bx)-out_min) / out_rng );
	    //gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * norm_val ) ) );
	    gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * (log(.01) - log(std::max(.01f,norm_val))) / (-log(.01)) )));
	  } else { gv = grey_to_pel( 0 ); }
	  out_img->set_pel( x, y, gv );
	}
      }
      out_img->save_fn_png( img_out_fn.exp );


    }

    p_Net_float init_caffe( string const & param_str ) {
      google::InitGoogleLogging("boda_caffe");
      Caffe::set_phase(Caffe::TEST);
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(0);
      //Caffe::set_mode(Caffe::CPU);

      NetParameter param;
      bool const ret = google::protobuf::TextFormat::ParseFromString( param_str, &param );
      assert_st( ret );

      p_Net_float net( new Net_float( param ) );
      net->CopyTrainedLayersFrom( trained_fn.exp );

#if 0
      int total_iter = 10; // atoi(argv[3]);
      LOG(ERROR) << "Running " << total_iter << " iterations.";

      double test_accuracy = 0;
      for (int i = 0; i < total_iter; ++i) {
	const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
	test_accuracy += result[0]->cpu_data()[0];
	LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
      }
      test_accuracy /= total_iter;
      LOG(ERROR) << "Test accuracy: " << test_accuracy;
#endif
      return net;
    }

  };

  void raw_do_forward( shared_ptr<Net<float> > net_, vect_p_nda_float_t const & bottom ) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    assert_st( bottom.size() == input_blobs.size() );
    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      assert_st( bottom[i]->elems.sz == uint32_t(input_blobs[i]->count()) );
      const float* const data_ptr = &bottom[i]->elems[0];
      switch (Caffe::mode()) {
      case Caffe::CPU:
	memcpy(input_blobs[i]->mutable_cpu_data(), data_ptr,
	       sizeof(float) * input_blobs[i]->count());
	break;
      case Caffe::GPU:
	cudaMemcpy(input_blobs[i]->mutable_gpu_data(), data_ptr,
		   sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
	break;
      default:
	rt_err( "Unknown Caffe mode." );
      }  // switch (Caffe::mode())
    }
    //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    net_->ForwardPrefilled();
  }

  uint32_t get_layer_ix( shared_ptr<Net<float> > net_, string const & out_layer_name ) {
    vect_string const & layer_names = net_->layer_names();
    for( uint32_t i = 0; i != layer_names.size(); ++i ) { if( out_layer_name == layer_names[i] ) { return i; } }
    rt_err( strprintf("layer out_layer_name=%s not found in netowrk\n",str(out_layer_name).c_str() )); 
  }

  void copy_output_blob_data( shared_ptr<Net<float> > net_, string const & out_layer_name,
				     vect_p_nda_float_t & top )
  {
    uint32_t const out_layer_ix = get_layer_ix( net_, out_layer_name );
    const vector<Blob<float>*>& output_blobs = net_->top_vecs()[ out_layer_ix ];
    top.clear();
    for( uint32_t bix = 0; bix < output_blobs.size(); ++bix ) {
      Blob<float> * const output_blob = output_blobs[bix];
      dims_t out_batch_dims( 4 );
      out_batch_dims.dims(3) = output_blob->width();
      out_batch_dims.dims(2) = output_blob->height();
      out_batch_dims.dims(1) = output_blob->channels();
      out_batch_dims.dims(0) = output_blob->num();
      p_nda_float_t out_batch( new nda_float_t );
      out_batch->set_dims( out_batch_dims );
      assert_st( out_batch->elems.sz == uint32_t(output_blob->count()) );
      top.push_back( out_batch );
      
      float * const dest = &out_batch->elems[0];
      switch (Caffe::mode()) {
      case Caffe::CPU: memcpy(dest, output_blob->cpu_data(), sizeof(float) * output_blob->count() ); break;
      case Caffe::GPU: cudaMemcpy(dest, output_blob->gpu_data(), sizeof(float) * output_blob->count(), 
				  cudaMemcpyDeviceToHost); break;
      default: LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }


#include"gen/conv_pyra.cc.nesi_gen.cc"
}
