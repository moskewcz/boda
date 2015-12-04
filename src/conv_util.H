#ifndef _CONV_UTIL_H_
#define _CONV_UTIL_H_

#include"conv_common.H"

namespace caffe { struct NetParameter; struct NetState; }

namespace boda 
{
  struct has_conv_fwd_t; typedef shared_ptr< has_conv_fwd_t > p_has_conv_fwd_t; 
  typedef caffe::NetParameter net_param_t;
  typedef shared_ptr< net_param_t > p_net_param_t;
  typedef caffe::NetState net_state_t;
  typedef shared_ptr< net_state_t > p_net_state_t;

  // note: we use caffe-compatible layer type strings here
  // raw list: Pooling Convolution ReLU Dropout LRN Accuracy Softmax SoftmaxWithLoss Data Concat InnerProduct Spreading
  struct conv_op_info_t {
    std::string type;
    u32_pt_t num_bots; // min,max
    u32_pt_t num_tops; // min,max
  };
  conv_op_info_t const Pooling_coi{ "Pooling", {1,1}, {1,1} };
  conv_op_info_t const Convolution_coi{ "Convolution", {1,1}, {1,1} }; // FIXME_EFB
  conv_op_info_t const ReLU_coi{ "ReLU", {1,1}, {1,1} };
  conv_op_info_t const Dropout_coi{ "Dropout", {1,1}, {1,1} };
  conv_op_info_t const LRN_coi{ "LRN", {1,1}, {1,1} };
  conv_op_info_t const Accuracy_coi{ "Accuracy", {1,1}, {1,1} };
  conv_op_info_t const Softmax_coi{ "Softmax", {1,1}, {1,1} };
  conv_op_info_t const SoftmaxWithLoss_coi{ "SoftmaxWithLoss", {2,2}, {2,2} }; // { prob, label } --> { prob_grad, loss }
  conv_op_info_t const Data_coi{ "Data", {0,0}, {1,1} }; // note: no inputs / source
  conv_op_info_t const Concat_coi{ "Concat", {1,uint32_t_const_max}, {1,1} };
  conv_op_info_t const InnerProduct_coi{ "InnerProduct", {1,1}, {1,1} };
  // backwards-specific layers. there might be better/more-common names for these (and we will change/update them as
  // makes sense), but the idea is that they are operations in thier own right, not just 'backwards' versions of some
  // other ops. so we try to understand what they do functionally and name them accordingly.
  conv_op_info_t const Spreading_coi{ "Spreading", {3,3}, {1,1} }; // { out, out_grad_loss, in } --> { in_grad_loss }
  conv_op_info_t const ZeroIfNeg_coi{ "ZeroIfNeg", {2,2}, {1,1} }; // { in, cond } --> { out } // note: dims(cond)==dims(out)==dims(in);out=(cond>=0)?in:0
  // FIXME_EFB: { in, filts, biases, out_grad_loss } --> { in_grad_loss, filts_grad_loss, biases_grad_loss }
  // otherwise: { in, out_grad_loss } --> { in_grad_loss }
  conv_op_info_t const BckConv_coi{ "BckConv", {2,2}, {1,1} }; 
  

  struct conv_op_t : virtual public nesi // NESI(help="conv_op descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string tag; //NESI(help="tag to refer to conv op by",req=1)
    string type; //NESI(help="type of op; may impact in/out size calculations",req=1)
    u32_box_t in_pad; //NESI(default="0 0 0 0",help="input padding")
    u32_pt_t kern_sz; //NESI(default="0 0",help="convolutional kernel size")
    u32_pt_t stride; //NESI(default="1 1",help="step/stride in input")

    // related to depth (currently unused; might be optional?)
    uint32_t out_chans; //NESI(default="0",help="number of output channels")
    // uint32_t groups; NESI(default="1",help="number of groups (equal partitions of inputs and outputs)")

    uint32_t avg_pool; //NESI(default="0",help="0 for max pooling, 1 for average pooling (others unsupported for compute)")

    uint32_t lrn_local_size; //NESI(default="5",help="LRN local_size param")
    double lrn_alpha; //NESI(default="1.0",help="LRN alpha param")
    double lrn_beta; //NESI(default="0.75",help="LRN beta param")
    double lrn_k; //NESI(default="1.0",help="LRN k param")

    vect_string tops; // inputs (by node/blob name)
    vect_string bots; // outputs (by node/blob name)

    u32_pt_t out_sz_to_in_sz( u32_pt_t const & out_sz, bool const ignore_padding ) const;
    u32_pt_t in_sz_to_out_sz( u32_pt_t const & in_sz, bool const ignore_padding ) const;

    // seen is a temportary to allow exactly *one* topo visit at a time. callers should use *exactly one of*
    // on_seen_bot()/on_seen_top() for a given traversal, depending on the traversal direction. FIXME: it would
    // perhaps be cleaner to use a map from string->seen for each traversal, and this would allow multiple concurrent
    // traversal. currently we don't seem to need that, though.
    uint32_t seen;
    bool on_seen_bot( void ) { ++seen; assert_st( seen <= bots.size() ); return seen == bots.size(); }
    bool on_seen_top( void ) { ++seen; assert_st( seen <= tops.size() ); return seen == tops.size(); }

    bool has_one_top( void ) const { return tops.size() == 1; }
    bool has_one_top_one_bot( void ) const { return (tops.size() == 1) && (bots.size() == 1); }
    string const & get_single_in_place_arg( void ) const { assert_st( has_one_top_one_bot() ); assert_st( tops[0] == bots[0] ); return tops[0]; }  

    //FIXME: for now, we default-init these vars to prevent them from being garbage when they are unused. this is
    //noticably bad in combination with always printing all parameters. we should do some combination of: (1) always
    //using NESI to init conv_op_t's (2) custom-printing conv_op_t's to avoid printing ununsed/invalid fields (3)
    //actually use some hierarchy here (gasp!)?
    conv_op_t( void ) : out_chans(0), avg_pool(0), lrn_local_size(0), lrn_alpha(0), lrn_beta(0), lrn_k(0) { }
    bool is( conv_op_info_t const & coi ) const; // type string checking + verify input/output argument count and other sanity checks
  };

  typedef vector< conv_op_t > vect_conv_op_t; 
  typedef shared_ptr< conv_op_t > p_conv_op_t; 
  typedef vector< p_conv_op_t > vect_p_conv_op_t;
  typedef shared_ptr< vect_conv_op_t > p_vect_conv_op_t; 
  typedef shared_ptr< vect_p_conv_op_t > p_vect_p_conv_op_t; 

  typedef map< string, p_conv_op_t > map_str_p_conv_op_t;
  typedef shared_ptr< map_str_p_conv_op_t > p_map_str_p_conv_op_t;

  struct conv_node_t {
    std::string name;
    dims_t dims;
    vect_string bot_for;
    vect_string top_for;
    conv_support_info_t csi;
    conv_io_t cio;
    vect_p_conv_op_t in_place_ops;
   conv_node_t( std::string const & name_ ) : name(name_) { }
  };

  typedef vector< conv_node_t > vect_conv_node_t; 
  typedef shared_ptr< conv_node_t > p_conv_node_t; 
  typedef vector< p_conv_node_t > vect_p_conv_node_t;
  typedef map< string, p_conv_node_t > map_str_p_conv_node_t;
  typedef shared_ptr< map_str_p_conv_node_t > p_map_str_p_conv_node_t;

  struct conv_pipe_t {
    string out_node_name;
    p_map_str_p_conv_op_t convs;
    p_map_str_p_conv_node_t nodes;
    // global top and bottom sets of nodes (sources and sinks of whole network)
    zi_uint32_t data_num_imgs;
    set_string tops;
    set_string bots;
    vect_string data_img_node_names;
    vect_string data_label_node_names;

    p_map_str_p_nda_float_t op_params; // layer blobs, as layer_BLOBIX->blob or 'pretty' names for some layer types like layer_filts->blob
    p_map_str_p_vect_p_nda_float_t layer_blobs; // same info as op_params, but in a generic layer_name->vect_blobs format

    conv_pipe_t( void ) : convs( new map_str_p_conv_op_t ), nodes( new map_str_p_conv_node_t ), 
			  op_params( new map_str_p_nda_float_t ), layer_blobs( new map_str_p_vect_p_nda_float_t ) { }

    p_conv_node_t get_or_make_node( std::string const & name, bool const is_bot, bool const is_top );
    p_conv_node_t must_get_node( std::string const & name ) const;
    p_conv_node_t get_fwd_top_for_label( string const & n ) const;
    p_conv_op_t get_op( string const & name ) const;
    dims_t get_data_img_dims( void ) const;
    u32_pt_t get_data_img_xy_dims_3_chans_only( void ) const;

    p_conv_node_t get_single_top_node( void ) const;
    p_conv_op_t maybe_get_single_writer( p_conv_node_t const & node ) const;
    p_conv_op_t get_single_writer( p_conv_node_t const & node ) const;
    p_conv_op_t maybe_get_single_parent( p_conv_op_t const & cop ) const;

    void add_conv( p_conv_op_t const & conv );

    void calc_dims_op( p_conv_op_t const & cop );
    void calc_dims_rec( string const & node_name );
    void calc_dims( void );

    void calc_support_forward_op( p_conv_op_t const & cop, bool const ignore_padding );
    void calc_support_forward_rec( string const & node_name, bool const ignore_padding );
    void calc_support_info( bool const ignore_padding );

    void topo_visit_setup( void );

    void clear_sizes( void );
    void calc_sizes_forward_op( p_conv_op_t const & cop, bool const ignore_padding );
    void calc_sizes_forward_rec( string const & node_in, bool const ignore_padding );
    void calc_sizes_forward( bool const ignore_padding );
    void calc_sizes_back_rec( p_conv_node_t const & node_out, bool const ignore_padding );
    void calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding );

    void dump_pipe_rec( std::ostream & out, string const & node_name );
    void dump_pipe( std::ostream & out );
    void dump_ios_rec( std::ostream & out, string const & node_name );
    void dump_ios( std::ostream & out );
    void dump_ops_rec( std::ostream & out, string const & node_name, bool const & expand_ops );
    void dump_ops( std::ostream & out, bool const & expand_ops );

    void add_bck_ops_op( p_conv_op_t const & cop );
    void add_bck_ops_rec( string const & node_name );
    void add_bck_ops( void );

    void fwd_alloc_ndas( p_map_str_p_nda_float_t const & fwd, uint32_t const & num_imgs, bool const & sinks_only );
    p_nda_float_t run_one_blob_in_one_blob_out( p_nda_float_t const & in, p_has_conv_fwd_t const & conv_fwd );

    void add_layer_blobs( string const & rln, p_vect_p_nda_float_t const & blobs );

    p_net_param_t as_net_param( void ) const; // return this pipe as a caffe net param
    // FIXME: we use orig_net_param to cheat on the implimentation of as_net_param() under the assumption that the
    // pipe is always created from a net_param, and that the net_param is substantially equivalent to the pipe. in
    // general neither need be true; pipes could come from other sources, and the translation from input
    // net_param_t's is limited/inexact.
    p_net_param_t orig_net_param;
    // extra 'cheating' vars, so caffe_fwd can 'emulate' bck ops (i.e. by doing regular caffe-style backward), and knows the net-state filter
    p_net_state_t net_state;
    zi_bool has_bck_ops; 

  };


  typedef vector< conv_pipe_t > vect_conv_pipe_t; 
  typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 
  typedef vector< p_conv_pipe_t > vect_p_conv_pipe_t;

}

#endif /* _CONV_UTIL_H_ */
