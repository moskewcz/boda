// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CAFFEIF_H_
#define _CAFFEIF_H_

#include"boda_base.H"
#include"has_main.H"
#include"geom_prim.H"
#include"caffe/caffe.hpp"

namespace boda 
{
  typedef caffe::Net< float > Net_float;
  typedef shared_ptr< Net_float > p_Net_float;
  p_Net_float init_caffe( string const & param_str, string const & trained_fn );
  void raw_do_forward( p_Net_float net_, vect_p_nda_float_t const & bottom );
  void copy_output_blob_data( p_Net_float net_, string const & out_layer_name, vect_p_nda_float_t & top );

  struct synset_elem_t;
  typedef vector< synset_elem_t > vect_synset_elem_t; 
  typedef shared_ptr< synset_elem_t > p_synset_elem_t; 
  typedef shared_ptr< vect_synset_elem_t > p_vect_synset_elem_t; 

  struct img_t;
  typedef shared_ptr< img_t > p_img_t; 

  struct run_cnet_t : virtual public nesi, public has_main_t // NESI(help="run a caffe net on a single input",
		     // bases=["has_main_t"], type_id="run_cnet")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t ptt_fn; //NESI(default="%(boda_test_dir)/run_cnet_imagenet_deploy.prototxt",help="input net prototxt template filename")
    filename_t trained_fn; //NESI(default="%(home_dir)/alexnet/alexnet_train_iter_470000_v1",help="input trained net from which to copy params")
    u32_pt_t in_sz; //NESI(default="227 227",help="input size to use for caffe net")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")

    filename_t img_in_fn; //NESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")

    string out_layer_name;//NESI(default="prob",help="output layer name of which to output top blob of")

    filename_t out_labels_fn; //NESI(default="%(caffe_dir)/data/ilsvrc12/synset_words.txt",help="list of labels in net output layer channel order")

    virtual void main( nesi_init_arg_t * nia );

    p_vect_synset_elem_t out_labels;
    p_nda_float_t in_batch;
    p_Net_float net;

    void setup_predict( void );
    void do_predict( p_img_t const & img_in_ds );
  };

  struct run_cnet_t; typedef shared_ptr< run_cnet_t > p_run_cnet_t; 

}

#endif /* _CAFFEIF_H_ */
