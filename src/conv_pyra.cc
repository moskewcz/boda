// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"

#include "caffe/caffe.hpp"
#include <glog/logging.h>

namespace boda 
{
  using namespace boost;
  using namespace caffe;
  void init_caffe( void );

  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",bases=["has_main_t"], type_id="conv_pyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t pipe_fn; //NESI(default="%(boda_test_dir)/conv_pyra_pipe.xml",help="input pipe XML filename")
    filename_t ptt_fn; //NESI(default="%(boda_test_dir)/conv_pyra_net.prototxt",help="input net prototxt template filename")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")

    virtual void main( nesi_init_arg_t * nia ) { 
      p_string ptt_str = read_whole_fn( ptt_fn );
      string out_pt_str;
      str_format_from_nvm_str( out_pt_str, *ptt_str, 
			       strprintf( "(xsize=%s,ysize=%s)", str(100).c_str(), str(105).c_str() ) );
      (*ofs_open( out_fn )) << out_pt_str;
      init_caffe();
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

  void init_caffe( void ) {
    google::InitGoogleLogging("boda_caffe");

    Caffe::set_phase(Caffe::TEST);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(1);
    //Caffe::set_mode(Caffe::CPU);

    Net<float> caffe_test_net("/home/moskewcz/git_work/caffe_3/examples/imagenet/imagenet_deploy.prototxt");
    caffe_test_net.CopyTrainedLayersFrom("/home/moskewcz/alexnet/alexnet_train_iter_470000_v1");

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

#include"gen/conv_pyra.cc.nesi_gen.cc"
}
