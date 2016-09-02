#ifndef _CONV_UTIL_H_
#define _CONV_UTIL_H_

#include"conv_common.H"
#include"op_base.H"
#include"str_util.H"
#include"rtc_compute.H"

namespace caffe { class NetParameter; class NetState; }

namespace boda 
{
  struct has_conv_fwd_t; typedef shared_ptr< has_conv_fwd_t > p_has_conv_fwd_t; 
  typedef caffe::NetParameter net_param_t;
  typedef shared_ptr< net_param_t > p_net_param_t;
  typedef caffe::NetState net_state_t;
  typedef shared_ptr< net_state_t > p_net_state_t;

  // note: we use caffe-compatible layer type strings here
  // raw list: Pooling Convolution ReLU Dropout LRN Accuracy Softmax SoftmaxWithLoss Data Concat InnerProduct Spreading ZeroIfNonPos BckConv

  struct conv_op_info_t {
    std::string type;
    // map_str_str str_vals; // note: a full/general str_vals isn't needed/used here, just a type field. FIXME? 
    vect_string bots; 
    vect_string tops;
    map_str_p_nda_t nda_vals;
    zi_bool has_var_bots;
    zi_bool has_var_tops;
    string bot_an( uint32_t const & ix ) const;
    string top_an( uint32_t const & ix ) const;
    void op_reset_type( op_base_t & op ) const { op.erase_type(); op.set_type( type ); }
  };
  extern conv_op_info_t const clone_coi;
  extern conv_op_info_t const sgemm_coi;
  extern conv_op_info_t const Pooling_coi;
  extern conv_op_info_t const Convolution_coi;
  extern conv_op_info_t const Deconvolution_coi;
  extern conv_op_info_t const ReLU_coi;
  extern conv_op_info_t const Scale_coi;
  extern conv_op_info_t const BatchNorm_coi;
  extern conv_op_info_t const Dropout_coi;
  extern conv_op_info_t const BckDropout_coi;
  extern conv_op_info_t const LRN_coi;
  extern conv_op_info_t const BckLRN_coi;
  extern conv_op_info_t const Accuracy_coi;
  extern conv_op_info_t const Softmax_coi;
  extern conv_op_info_t const SoftmaxWithLoss_coi;
  extern conv_op_info_t const Data_coi;
  extern conv_op_info_t const Concat_coi;
  extern conv_op_info_t const Reduce_coi;
  extern conv_op_info_t const Eltwise_coi;
  extern conv_op_info_t const Split_coi;
  extern conv_op_info_t const InnerProduct_coi;
  extern conv_op_info_t const Spreading_coi;
  extern conv_op_info_t const ZeroIfNonPos_coi;
  extern conv_op_info_t const BckConv_coi;

  
  typedef conv_op_info_t const * rp_conv_op_info_t; 
  typedef vector< rp_conv_op_info_t > vect_rp_conv_op_info_t;  


  struct conv_pipe_t;
  typedef vector< conv_pipe_t > vect_conv_pipe_t; 
  typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 
  typedef vector< p_conv_pipe_t > vect_p_conv_pipe_t;

  // conv_opt_base_t: adds the coi field, which defines cnn-operation-type-specific information for this operation. In
  // partiuclar, it has information on the valid values for the 'type' string, and for each valid type, information on
  // the number and type of arguments for the operation and any other paramters.  this information is purely an
  // annotation; there is no added information that changes the meaning of the operation defined by the type and vals in
  // the base class fields.
  struct conv_op_base_t : public op_base_t // NESI(help="conv_op descriptor",bases=["op_base_t"])
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    conv_op_info_t const * coi;
    void set_and_check_coi( void ); // verify that 'type' string is recognized and set coi field accordingly to matching info struct
    bool is( conv_op_info_t const & coi ) const; // type string checking 
    conv_op_base_t( void ) : coi(0) { }

    // convienince accessor functions for commonly-used cnn operation fields
    u32_pt_t in_pad( void ) const { return get_xy_dims_strict( get_dims( "in_pad" ) ); }
    u32_pt_t kern_sz( void ) const { return get_xy_dims_strict( get_dims( "kern_sz" ) ); }
    u32_pt_t stride( void ) const { return get_xy_dims_strict( get_dims( "stride" ) ); }
    // FIXME: these fields are only used codegen, so maybe they don't belong here?
  };

  // conv_opt_t: adds information to an operation about where how it relates to other operations (i.e what it's inputs
  // and outputs actually are in a global context; stored in tops/bots/arg_map) and a global name for this particular
  // instance of the underlying operation (tag). this added information does not change the semantics of the operation
  // defined by the conv_op_base_t part.  FIXME: the in_place and seen fields really don't belong here ...
  struct conv_op_t : public conv_op_base_t // NESI(help="conv_op descriptor",bases=["conv_op_base_t"])
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string tag; //NESI(help="tag to refer to conv op by",req=1)

    vect_string tops; // inputs (by node/blob name)
    vect_string bots; // outputs (by node/blob name)
    zi_bool in_place; // set to 1 in add_conv() is the op in in_place (and omitted from top_for/bot_for lists)

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

    void set_and_check_coi_and_args( void ); // verify input/output argument count and other sanity checks

    // arg-map and related arg/dims handling -- generally for use after conv_pipe setup complete (i.e. in rtc)
    map_str_rtc_arg_t arg_map; // map from func arg names to call-site arg names (in this case, just in global/rtc scope)
    void set_arg_dims_and_map_from_pipe( conv_pipe_t const * const cp ); // add arg_name->dims mappings to dims_vals
    string get_arg( string const & an ) { rtc_arg_t const & arg = must_find( arg_map, an ); return arg.get_var(); }
    void set_arg( dims_t const & dims, string const & an, string const & vn ) {
      set_dims( an, dims );
      must_insert( arg_map, an, vn );
    }
    void reset_arg( string const & an, string const & vn ) { must_replace( arg_map, an, vn ); }
    void reset_arg_dims( string const & an, dims_t const & dims ) { reset_dims( an, dims ); }
    // FIXME: for now, null args are only distinguised by not passing a var at call-time. however, since most arguments
    // are non-optional, this is not ideal, since we can't error check not passing a var for most args. for now, we
    // preserve this set_null_arg_dims() function to mark the few cases where the arg is optional, pending a better
    // solution. note: these calls used to be set_null_arg(), which didn't need any dims, and was specially handled.
    void set_null_arg_dims( string const & an, dims_t const & dims ) { 
      set_dims( an, dims ); 
      must_insert( arg_map, an, make_dims_nda(dims) ); // note: the only place an is_nda() rtc_arg is put in arg_map ...
    }
    void erase_arg( string const & an ) { erase( an ); must_erase( arg_map, an ); }
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

    void calc_sizes_back_rec( p_conv_node_t const & node_out, bool const ignore_padding );
    void calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding );

    void dump_pipe_rec( std::ostream & out, string const & node_name );
    void dump_pipe( std::ostream & out );
    void dump_ios_rec( std::ostream & out, string const & node_name );
    void dump_ios( std::ostream & out );
    void dump_ops_rec( std::ostream & out, string const & node_name );
    void dump_ops( std::ostream & out );

    string get_grad_loss_onn( p_conv_op_t const & cop, string const & inn );
    void add_bck_ops_op( vect_p_conv_op_t & bck_ops, p_conv_op_t const & cop );
    void add_bck_ops_rec( vect_p_conv_op_t & bck_ops, string const & node_name );
    void add_bck_ops( void );

    void get_topo_order_caffe_comp_nodes( vect_string & out );

    void fwd_alloc_ndas( p_map_str_p_nda_float_t const & fwd, bool const & sinks_only );
    void run_setup_input( p_nda_float_t const & in, p_map_str_p_nda_float_t const & fwd, vect_string & in_vns );
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

}

#endif /* _CONV_UTIL_H_ */
