// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CAFFEIF_H_
#define _CAFFEIF_H_

#include"boda_base.H"
#include"has_main.H"
#include"has_conv_fwd.H"
#include"geom_prim.H"
#include"caffepb.H"

namespace boda 
{
  void subtract_mean_and_copy_img_to_batch( p_nda_float_t const & in_batch, uint32_t img_ix, p_img_t const & img );
  void copy_batch_to_img( p_nda_float_t const & out_batch, uint32_t img_ix, p_img_t const & img, u32_box_t region, 
			  float const & out_max  );
  void copy_batch_to_img( p_nda_float_t const & out_batch, uint32_t img_ix, p_img_t const & img, u32_box_t region );

  struct synset_elem_t;
  typedef vector< synset_elem_t > vect_synset_elem_t; 
  typedef shared_ptr< synset_elem_t > p_synset_elem_t; 
  typedef shared_ptr< vect_synset_elem_t > p_vect_synset_elem_t; 

  struct img_t; typedef shared_ptr< img_t > p_img_t; 
  struct conv_pipe_t; typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 

  struct conv_support_info_t;

  struct conv_io_t;
  typedef vector< conv_io_t > vect_conv_io_t; 
  typedef shared_ptr< conv_io_t > p_conv_io_t; 
  typedef vector< p_conv_io_t > vect_p_conv_io_t;
  typedef shared_ptr< vect_conv_io_t > p_vect_conv_io_t; 


  struct run_cnet_t : virtual public nesi, public has_main_t // NESI(help="run a caffe net on a single input",
		     // bases=["has_main_t"], type_id="run_cnet")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t ptt_fn; //NESI(default="%(models_dir)/%(model_name)/train_val.prototxt",help="input net prototxt template filename")
    filename_t trained_fn; //NESI(default="%(models_dir)/%(model_name)/best.caffemodel",help="input trained net from which to copy params")
    filename_t alt_trained_fn; //NESI(default="%(alt_param_dir)/%(model_name)/best.caffemodel",help="alternate location for input trained net from which to copy params (if trained_fn does not exist)")
    map_str_uint32_t in_dims; //NESI(default="()",help="input dims. any non-present dims will use values derived from the (single) data layer.")
    string out_node_name;//NESI(default="",help="output layer name of which to output top blob of")
    uint32_t anno_mode; //NESI(default=1,help="annotation string mode: "
    //" <default>0=label plus probs"
    //" 1=label only"
    //" 2=label plus sz"
    //" 3=label plus rank")
    uint32_t enable_upsamp_net; //NESI(default=0,help="if 1, create an extra internal net with internal 2X input upsampling.")
    uint32_t dump_rps; //NESI(default=0,help="if 1, dump region proposals for each feature pizel to 'rps.txt'.")
    p_has_conv_fwd_t conv_fwd; //NESI(default="(mode=caffe)",help="fwd compute mode")
    p_has_conv_fwd_t conv_fwd_upsamp; //NESI(default="(mode=caffe)",help="fwd compute mode (for upsampled net)")

    uint32_t add_bck_ops; //NESI(default=0,help="if non-zero, add bck (aka backwards/backprop/gradients) operations.")

    virtual void main( nesi_init_arg_t * nia );
    p_nda_float_t in_batch;

    void setup_cnet( void );

    p_nda_float_t run_one_blob_in_one_blob_out( void );
    p_nda_float_t run_one_blob_in_one_blob_out_upsamp( void );

    p_conv_pipe_t conv_pipe;
    p_conv_pipe_t conv_pipe_upsamp;
    conv_support_info_t const & get_out_csi( bool const & from_upsamp_net );
    dims_t const & get_out_dims( bool const & from_upsamp_net );
    u32_pt_t get_out_sz( bool const & from_upsamp_net ) { return get_xy_dims( get_out_dims(from_upsamp_net) ); }
    uint32_t get_out_chans( bool const & from_upsamp_net ) { return get_out_dims(from_upsamp_net).dsz("chan"); }
    uint32_t get_ceil_sqrt_out_chans( bool const & from_upsamp_net ) { return u32_ceil_sqrt( get_out_chans(from_upsamp_net) ); }

    u32_pt_t get_one_blob_img_out_sz( void ) { return get_out_sz(0).scale( u32_ceil_sqrt( get_out_chans(0) ) ); } // assumes/includes chans_to_area conv.

    //uint32_t out_s;
    run_cnet_t( void ) { }
    p_net_param_t net_param;
  private:
    p_net_param_t upsamp_net_param;
  };
  struct run_cnet_t; typedef shared_ptr< run_cnet_t > p_run_cnet_t; 

  struct anno_t;
  typedef vector< anno_t > vect_anno_t; 
  typedef shared_ptr< vect_anno_t > p_vect_anno_t; 

  struct scale_info_t {
    u32_pt_t img_sz; // for this scale. often used as (some_box * nominal_in_sz / img_sz) for scaling

    bool from_upsamp_net; // if true, use output of upsampled net
    uint32_t bix; // plane index
    u32_pt_t place; // offset of nc of scaled img inside plane

    i32_box_t feat_box;
    i32_box_t feat_img_box; // currently always == feat scaled up by out_s

    uint32_t psb; // starting index into pred_state. then index by chan,y,x (as per out_batch order)
    uint32_t get_psix( uint32_t const & c, u32_pt_t const & xy ) const { // xy in in place space
      assert_st( feat_box.strictly_contains( u32_to_i32(xy) ) );
      u32_pt_t const off = xy - i32_to_u32(feat_box.p[0]);
      return psb + c*feat_box.sz().dims_prod() + off.d[1]*feat_box.sz().d[0] + off.d[0]; }
  };

  typedef vector< scale_info_t > vect_scale_info_t; 
  std::ostream & operator <<(std::ostream & os, scale_info_t const & v ); 

  struct pred_state_t {
    double cur_prob;
    double filt_prob;
    bool to_disp;
    bool filt_prob_init;
    uint32_t label_ix;
    i32_box_t img_box;
    pred_state_t( void ) : cur_prob(0), filt_prob(0), to_disp(0), filt_prob_init(0), label_ix(0) { }
  };
  typedef vector< pred_state_t > vect_pred_state_t; 

  typedef map_i32_pt_t_to_i32_box_t nms_grid_t;

  struct cnet_predict_t : virtual public nesi, public run_cnet_t // NESI(help="run a caffe net on a single input",
		     // bases=["run_cnet_t"], type_id="cnet_predict")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")
    filename_t img_in_fn; //NESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")

    filename_t out_labels_fn; //NESI(default="%(caffe_dir)/data/ilsvrc12/synset_words.txt",help="list of labels in net output layer channel order")
    double filt_rate;//NESI(default=".2",help="filter rate cooef labels display. should be < 1.0.")
    double filt_show_thresh;//NESI(default=".02",help="start display of labels above this filtered prob. should be < 1.0 and > filt_drop_thresh.")
    double filt_drop_thresh;//NESI(default=".005",help="drop display of labels below this filtered prob. should be < 1.0 and < filt_show_thresh.")

    uint32_t max_num_disp; //NESI(default="20",help="max # of results to display total (i.e. TOP_N)")
    uint32_t max_labels_per_anno; //NESI(default="5",help="max # of labels to display per-anno (i.e. TOP_N)")
    u32_pt_t nms_grid_pels; //NESI(default="25 25",help="nms coverage grid precision")
    double nms_core_rat; //NESI(default=".75",help="multiply box by this factor to determine nms converage area")


    virtual void main( nesi_init_arg_t * nia );

    p_vect_synset_elem_t out_labels;
    vect_scale_info_t scale_infos;
    u32_pt_t nominal_in_sz;
    vect_pred_state_t pred_state;

    void setup_predict( void );
    void setup_scale_infos( void );
    void setup_scale_infos( uint32_t const & interval, vect_u32_pt_t const & sizes, 
			    vect_u32_pt_w_t const & placements, u32_pt_t const & nominal_in_sz_ );

    p_vect_anno_t do_predict( p_img_t const & img_in_ds, bool const print_to_terminal );
    p_vect_anno_t do_predict( p_nda_float_t const & out_batch, p_nda_float_t const & out_batch_upsamp, 
			      bool const print_to_terminal );
    void do_predict_region( p_nda_float_t const & out_batch, dims_t const & obb, dims_t const & obe, 
			    uint32_t const & psb );
    p_vect_anno_t pred_state_to_annos( bool const print_to_terminal );
  private:
    nms_grid_t nms_grid;
    i32_box_t nms_grid_op( bool const & do_set, i32_box_t const & img_box );
  };

  struct cnet_predict_t; typedef shared_ptr< cnet_predict_t > p_cnet_predict_t; 

}

#endif /* _CAFFEIF_H_ */
