#ifndef _CONV_UTIL_H_
#define _CONV_UTIL_H_

#include"conv_common.H"

namespace boda 
{
  struct has_conv_fwd_t; typedef shared_ptr< has_conv_fwd_t > p_has_conv_fwd_t; 

  // caffe-compatible layer type strings
  // raw list: Pooling Convolution ReLU Dropout LRN Accuracy SoftmaxWithLoss Data Concat InnerProduct
  std::string const Pooling_str = "Pooling";
  std::string const Convolution_str = "Convolution";
  std::string const ReLU_str = "ReLU";
  std::string const Dropout_str = "Dropout";
  std::string const LRN_str = "LRN";
  std::string const Accuracy_str = "Accuracy";
  std::string const Softmax_str = "Softmax";
  std::string const SoftmaxWithLoss_str = "SoftmaxWithLoss";
  std::string const Data_str = "Data";
  std::string const Concat_str = "Concat";
  std::string const InnerProduct_str = "InnerProduct";

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

    uint32_t bots_seen;
    bool on_seen_bot( void ) { ++bots_seen; assert_st( bots_seen <= bots.size() ); return bots_seen == bots.size(); }
    bool has_one_top( void ) const { return tops.size() == 1; }
    bool has_one_top_one_bot( void ) const { return (tops.size() == 1) && (bots.size() == 1); }
    string const & get_single_in_place_arg( void ) const { assert_st( has_one_top_one_bot() ); assert_st( tops[0] == bots[0] ); return tops[0]; }  

    //FIXME: for now, we default-init these vars to prevent them from being garbage when they are unused. this is
    //noticably bad in combination with always printing all parameters. we should do some combination of: (1) always
    //using NESI to init conv_op_t's (2) custom-printing conv_op_t's to avoid printing ununsed/invalid fields (3)
    //actually use some hierarchy here (gasp!)?
    conv_op_t( void ) : out_chans(0), avg_pool(0), lrn_local_size(0), lrn_alpha(0), lrn_beta(0), lrn_k(0) { }
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
    string out_layer_name;
    p_map_str_p_conv_op_t convs;
    p_map_str_p_conv_node_t nodes;
    // global top and bottom sets of nodes (sources and sinks of whole network)
    vect_string tops;
    vect_string bots;

    p_map_str_p_nda_float_t op_params;
    bool finalized;

    conv_pipe_t( void ) : convs( new map_str_p_conv_op_t ), nodes( new map_str_p_conv_node_t ), 
			  op_params( new map_str_p_nda_float_t ), finalized(0) { }
    void finalize( void ); // call when finished building network
    p_conv_node_t get_or_make_node( std::string const & name );
    p_conv_node_t must_get_node( std::string const & name ) const;
    p_conv_op_t get_op( string const & name ) const;
    p_conv_node_t get_single_bot_node( void ) const;
    p_conv_node_t get_single_top_node( void ) const;
    p_conv_op_t maybe_get_single_writer( p_conv_node_t const & node ) const;
    p_conv_op_t maybe_get_single_parent( p_conv_op_t const & cop ) const;

    void add_conv( p_conv_op_t const & conv );
    void calc_support_forward_rec( p_conv_node_t const & node_in, bool const ignore_padding );
    void calc_support_info( bool const ignore_padding, uint32_t const & in_chans );

    void topo_visit_setup( void );

    void clear_sizes( void );
    void calc_sizes_forward_rec( p_conv_node_t const & node_in, bool const ignore_padding );
    void calc_sizes_forward( u32_pt_t const & in_sz, bool const ignore_padding );
    void calc_sizes_back_rec( p_conv_node_t const & node_out, bool const ignore_padding );
    void calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding );

    void dump_pipe_rec( std::ostream & out, string const & node_name );
    void dump_pipe( std::ostream & out );
    void dump_ios_rec( std::ostream & out, string const & node_name );
    void dump_ios( std::ostream & out );
    void dump_ops_rec( std::ostream & out, string const & node_name, bool const & expand_ops );
    void dump_ops( std::ostream & out, bool const & expand_ops );

    void fwd_alloc_ndas( p_map_str_p_nda_float_t const & fwd, uint32_t const & num_imgs, bool const & sinks_only );
    p_nda_float_t run_one_blob_in_one_blob_out( p_nda_float_t const & in, p_has_conv_fwd_t const & conv_fwd );
  };


  typedef vector< conv_pipe_t > vect_conv_pipe_t; 
  typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 
  typedef vector< p_conv_pipe_t > vect_p_conv_pipe_t;

}

#endif /* _CONV_UTIL_H_ */
