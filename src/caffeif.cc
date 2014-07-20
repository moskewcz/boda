// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include"str_util.H"
#include"img_io.H"
#include"lexp.H"

#include "caffeif.H"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

namespace boda 
{
  using namespace boost;
  using caffe::Caffe;
  using caffe::Blob;

  p_Net_float init_caffe( string const & param_str, string const & trained_fn ) {
    timer_t t("caffe_init");
    google::InitGoogleLogging("boda_caffe");
    Caffe::set_phase(Caffe::TEST);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);
    //Caffe::set_mode(Caffe::CPU);

    caffe::NetParameter param;
    bool const ret = google::protobuf::TextFormat::ParseFromString( param_str, &param );
    assert_st( ret );

    p_Net_float net( new Net_float( param ) );
    net->CopyTrainedLayersFrom( trained_fn );

#if 0
    int total_iter = 10; // atoi(argv[3]);
    LOG(ERROR) << "Running " << total_iter << " iterations.";

    double test_accuracy = 0;
    for (int i = 0; i < total_iter; ++i) {
      const vector<Blob<float>*>& result = net->ForwardPrefilled();
      test_accuracy += result[0]->cpu_data()[0];
      LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
    }
    test_accuracy /= total_iter;
    LOG(ERROR) << "Test accuracy: " << test_accuracy;
#endif
    return net;
  }


  void raw_do_forward( p_Net_float net_, vect_p_nda_float_t const & bottom ) {
    timer_t t("caffe_forward");
    vector<caffe::Blob<float>*>& input_blobs = net_->input_blobs();
    assert_st( bottom.size() == input_blobs.size() );
    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      assert_st( bottom[i]->elems.sz == uint32_t(input_blobs[i]->count()) );
      const float* const data_ptr = &bottom[i]->elems[0];
      switch ( Caffe::mode() ) {
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

  uint32_t get_layer_ix( p_Net_float net_, string const & out_layer_name ) {
    vect_string const & layer_names = net_->layer_names();
    for( uint32_t i = 0; i != layer_names.size(); ++i ) { if( out_layer_name == layer_names[i] ) { return i; } }
    rt_err( strprintf("layer out_layer_name=%s not found in netowrk\n",str(out_layer_name).c_str() )); 
  }

  void copy_output_blob_data( p_Net_float net_, string const & out_layer_name, vect_p_nda_float_t & top )
  {
    timer_t t("caffe_copy_output_blob_data");
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

  struct synset_elem_t {
    string id;
    string tag;
  };
  std::ostream & operator <<(std::ostream & os, synset_elem_t const & v) { return os << "id=" << v.id << " tag=" << v.tag; }
  

  void read_synset( p_vect_synset_elem_t out, filename_t const & fn ) {
    p_ifstream ifs = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn.in, ifs, line ) ) {
      size_t const spos = line.find( ' ' );
      if( spos == string::npos ) { rt_err( "failing to parse synset line: no space found: line was '" + line + "'" ); }
      size_t cpos = line.find( ',', spos+1 );
      if( cpos == string::npos ) { cpos = line.size(); }
      assert( spos < cpos );
      uint32_t const tag_len = cpos - spos - 1;
      if( !tag_len ) { rt_err( "failing to parse synset line: no tag found, (comma after first space) or (first space at end of line (note: implies no command after first space)): line was '" + line + "'" ); }
      synset_elem_t t;
      t.id = string( line, 0, spos );
      t.tag = string( line, spos+1, tag_len );
      out->push_back( t );
    }
    //printf( "out=%s\n", str(out).c_str() );
  }


  void run_cnet_t::main( nesi_init_arg_t * nia ) { 
    setup_predict();
    p_img_t img_in( new img_t );
    img_in->load_fn( img_in_fn.exp );
    p_img_t img_in_ds = downsample_to_size( downsample_2x_to_size( img_in, in_sz.d[0], in_sz.d[1] ), 
					    in_sz.d[0], in_sz.d[1] );
    do_predict( img_in_ds );
  }


  void run_cnet_t::setup_predict( void ) {
    assert_st( !out_labels );
    out_labels.reset( new vect_synset_elem_t );
    read_synset( out_labels, out_labels_fn );

    p_string ptt_str = read_whole_fn( ptt_fn );
    string out_pt_str;
    str_format_from_nvm_str( out_pt_str, *ptt_str, 
			     strprintf( "(xsize=%s,ysize=%s,num=%s,chan=%s)", 
					str(in_sz.d[0]).c_str(), str(in_sz.d[1]).c_str(), 
					str(1).c_str(), str(3).c_str() ) );
    assert( !net );
    net = init_caffe( out_pt_str, trained_fn.exp );      

    dims_t in_batch_dims( 4 );
    in_batch_dims.dims(3) = in_sz.d[0];
    in_batch_dims.dims(2) = in_sz.d[1];
    in_batch_dims.dims(1) = 3; 
    in_batch_dims.dims(0) = 1;
     
    assert( !in_batch );
    in_batch.reset( new nda_float_t );
    in_batch->set_dims( in_batch_dims );
  }

  void run_cnet_t::do_predict( p_img_t const & img_in_ds ) {
    assert( img_in_ds->w == in_sz.d[0] );
    assert( img_in_ds->h == in_sz.d[1] );
    uint32_t const inmc = 123U+(117U<<8)+(104U<<16)+(255U<<24); // RGBA
    {
#pragma omp parallel for	  
      for( uint32_t y = 0; y < in_sz.d[1]; ++y ) {
	for( uint32_t x = 0; x < in_sz.d[0]; ++x ) {
	  uint32_t const pel = img_in_ds->get_pel(x,y);
	  for( uint32_t c = 0; c < 3; ++c ) {
	    in_batch->at4(0,2-c,y,x) = get_chan(c,pel) - float(uint8_t(inmc >> (c*8)));
	  }
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
    assert( obd.dims(0) == 1 );
    assert( obd.dims(1) == out_labels->size() );
    assert( obd.dims(2) == 1 );
    assert( obd.dims(3) == 1 );

    for( uint32_t i = 0; i < obd.dims(1); ++i ) {
      float const p = out_batch->at4(0,i,0,0);
      if( p >= .01 ) {
	printf( "out_labels->at(i)=%s p=%s\n", str(out_labels->at(i)).c_str(), str(p).c_str() );
      }
    }
    //printf( "obd=%s\n", str(obd).c_str() );
    //(*ofs_open( out_fn )) << out_pt_str;
  }

#include"gen/caffeif.H.nesi_gen.cc"
}

