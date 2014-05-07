// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"conv_util.H"

#include "caffe/caffe.hpp"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

namespace boda 
{
  using namespace boost;
  using namespace caffe;
  void init_caffe( string const & param_str );

  p_conv_pipe_t make_p_conv_pipe_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );
  
  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",bases=["has_main_t"], type_id="conv_pyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t pipe_fn; //NESI(default="%(boda_test_dir)/conv_pyra_pipe.xml",help="input pipe XML filename")
    filename_t ptt_fn; //NESI(default="%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt",help="input net prototxt template filename")
    filename_t trained_fn; //NESI(default="%(home_dir)/alexnet/alexnet_train_iter_470000_v1",help="input trained net from which to copy params")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")

    virtual void main( nesi_init_arg_t * nia ) { 
      p_conv_pipe_t conv_pipe = make_p_conv_pipe_t_init_and_check_unused_from_lexp( parse_lexp_xml_file( pipe_fn.exp ), 0 );

      p_string ptt_str = read_whole_fn( ptt_fn );
      string out_pt_str;
      str_format_from_nvm_str( out_pt_str, *ptt_str, 
			       strprintf( "(xsize=%s,ysize=%s,num=%s,chan=%s)", 
					  str(100).c_str(), str(100).c_str(), str(1).c_str(), str(3).c_str() ) );
      //(*ofs_open( out_fn )) << out_pt_str;
      init_caffe( out_pt_str );      

      
    }

    void init_caffe( string const & param_str ) {
      google::InitGoogleLogging("boda_caffe");
      Caffe::set_phase(Caffe::TEST);
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(1);
      //Caffe::set_mode(Caffe::CPU);

      NetParameter param;
      bool const ret = google::protobuf::TextFormat::ParseFromString( param_str, &param );
      assert_st( ret );

      Net<float> caffe_test_net( param );
      caffe_test_net.CopyTrainedLayersFrom( trained_fn.exp );

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
    }

  };

  static void raw_do_forward( shared_ptr<Net<float> > net_, vect_p_nda_float_t const & bottom ) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(bottom.size(), input_blobs.size());
    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      assert( bottom[i]->elems.sz == uint32_t(input_blobs[i]->count()) );
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
	LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    net_->ForwardPrefilled();
  }


#include"gen/conv_pyra.cc.nesi_gen.cc"
}
