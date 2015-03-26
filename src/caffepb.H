#ifndef _CAFFEPB_H_
#define _CAFFEPB_H_

// this file is meant to contain functionality that needs the caffe
// protobuf header but not the rest of caffe. thus, boda modes that
// only need this subset of caffe related functionality can function
// without a full caffe install/dependency. also, it should be
// possible to package only the caffe protobuf header and use it here.

#include<string>
#include"conv_util.H" // could use pre-decls instead ...

namespace caffe { struct NetParameter; }

namespace boda {
  typedef shared_ptr< caffe::NetParameter > p_net_param_t;
  p_conv_pipe_t create_pipe_from_param( caffe::NetParameter & net_param, std::string const & out_layer_name );
  template< typename CP > void set_param_from_conv_op( CP & cp, p_conv_op_t conv_op );
  template< typename CP > p_conv_op_t get_conv_op_from_param( CP const & cp );
  p_net_param_t parse_and_upgrade_net_param_from_text_file( filename_t const & ptt_fn );

}

#endif /* _CAFFEPB_H_ */
