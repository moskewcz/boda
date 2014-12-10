// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CAFFEIF_H_
#define _CAFFEIF_H_

#include"boda_base.H"
#include"has_main.H"
#include"geom_prim.H"

namespace caffe
{
  template< typename T > struct Net;
}

namespace boda 
{
  typedef caffe::Net< float > Net_float;
  typedef shared_ptr< Net_float > p_Net_float;
  p_Net_float init_caffe( string const & param_str, string const & trained_fn );
  void raw_do_forward( p_Net_float net_, vect_p_nda_float_t const & bottom );
  void copy_output_blob_data( p_Net_float net_, string const & out_layer_name, vect_p_nda_float_t & top );
  p_nda_float_t run_one_blob_in_one_blob_out( p_Net_float net, string const & out_layer_name, p_nda_float_t const & in );
  void subtract_mean_and_copy_img_to_batch( p_nda_float_t const & in_batch, uint32_t img_ix, p_img_t const & img );
  void copy_batch_to_img( p_nda_float_t const & out_batch, uint32_t img_ix, p_img_t const & img );

  struct synset_elem_t;
  typedef vector< synset_elem_t > vect_synset_elem_t; 
  typedef shared_ptr< synset_elem_t > p_synset_elem_t; 
  typedef shared_ptr< vect_synset_elem_t > p_vect_synset_elem_t; 

  struct img_t; typedef shared_ptr< img_t > p_img_t; 
  struct conv_pipe_t; typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 

  struct run_cnet_t : virtual public nesi, public has_main_t // NESI(help="run a caffe net on a single input",
		     // bases=["has_main_t"], type_id="run_cnet")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t ptt_fn; //NESI(default="%(boda_test_dir)/run_cnet_imagenet_deploy.prototxt",help="input net prototxt template filename")
    filename_t trained_fn; //NESI(default="%(models_dir)/alexnet/alexnet_train_iter_470000_v1",help="input trained net from which to copy params")
    u32_pt_t in_sz; //NESI(default="227 227",help="input size to use for caffe net")
    uint32_t in_num_imgs; //NESI(default="1",help="number of images to use for caffe net input layer/blob")
    uint32_t in_num_chans; //NESI(default="3",help="number of channels to use for caffe net input layer/blob")
    string out_layer_name;//NESI(default="prob",help="output layer name of which to output top blob of")
    virtual void main( nesi_init_arg_t * nia );
    p_nda_float_t in_batch;
    p_Net_float net;
    void setup_cnet( void );
    u32_pt_t get_one_blob_img_out_sz( void ); // note: assumes/includes chans_to_area conversion
    p_nda_float_t run_one_blob_in_one_blob_out( void );
    p_conv_pipe_t get_pipe( void );
  };
  struct run_cnet_t; typedef shared_ptr< run_cnet_t > p_run_cnet_t; 

  struct anno_t;
  typedef vector< anno_t > vect_anno_t; 
  typedef shared_ptr< vect_anno_t > p_vect_anno_t; 

  struct pred_state_t {
    float cur_prob;
    float filt_prob;
    bool to_disp;
    uint32_t label_ix;
  };
  typedef vector< pred_state_t > vect_pred_state_t; 

  struct cnet_predict_t : virtual public nesi, public run_cnet_t // NESI(help="run a caffe net on a single input",
		     // bases=["run_cnet_t"], type_id="cnet_predict")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")
    filename_t img_in_fn; //NESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")

    filename_t out_labels_fn; //NESI(default="%(caffe_dir)/data/ilsvrc12/synset_words.txt",help="list of labels in net output layer channel order")
    double filt_rate;//NESI(default=".02",help="filter rate cooef labels display. should be < 1.0.")
    double filt_show_thresh;//NESI(default=".02",help="start display of labels above this filtered prob. should be < 1.0 and > filt_drop_thresh.")
    double filt_drop_thresh;//NESI(default=".005",help="drop display of labels below this filtered prob. should be < 1.0 and < filt_show_thresh.")

    virtual void main( nesi_init_arg_t * nia );

    p_vect_synset_elem_t out_labels;

    vect_pred_state_t pred_state;

    void setup_predict( void );
    p_vect_anno_t do_predict( p_img_t const & img_in_ds, bool const print_to_terminal );
  };

  struct cnet_predict_t; typedef shared_ptr< cnet_predict_t > p_cnet_predict_t; 

  p_conv_pipe_t get_pipe( p_Net_float net, string const & out_layer_name );


}

#endif /* _CAFFEIF_H_ */
