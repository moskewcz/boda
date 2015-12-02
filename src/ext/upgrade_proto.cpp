// from the caffe framework, BSD licenced, see https://github.com/BVLC/caffe -- copied 2015.03.26
// modifications Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE

#include"../boda_tu_base.H"
#include"../str_util.H"
using boda::rt_err;
using boda::strprintf;
using boda::str;
#include "../gen/caffe.pb.h"
#include <map>
#include <string>

using std::map;
using std::string;


namespace boda_caffe {

  using namespace caffe;

void unk_param_err( string const & pname, string const & ltype ) { 
  rt_err( strprintf( "Unknown parameter %s for layer type %s", pname.c_str(), ltype.c_str() ) );
}

bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).has_layer()) {
      return true;
    }
  }
  return false;
}

bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param) {
  return net_param.layers_size() > 0;
}

bool NetNeedsUpgrade(const NetParameter& net_param) {
  return NetNeedsV0ToV1Upgrade(net_param) || NetNeedsV1ToV2Upgrade(net_param);
}

void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad) {
  // Copy everything other than the layers from the original param.
  param_upgraded_pad->Clear();
  param_upgraded_pad->CopyFrom(param);
  param_upgraded_pad->clear_layers();
  // Figure out which layer each bottom blob comes from.
  map<string, int> blob_name_to_last_top_idx;
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    blob_name_to_last_top_idx[blob_name] = -1;
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    const V1LayerParameter& layer_connection = param.layers(i);
    const V0LayerParameter& layer_param = layer_connection.layer();
    // Add the layer to the new net, unless it's a padding layer.
    if (layer_param.type() != "padding") {
      param_upgraded_pad->add_layers()->CopyFrom(layer_connection);
    }
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        rt_err( strprintf( "Unknown blob input %s to layer %s", blob_name.c_str(), str(j).c_str() ) );
      }
      const int top_idx = blob_name_to_last_top_idx[blob_name];
      if (top_idx == -1) {
        continue;
      }
      const V1LayerParameter& source_layer = param.layers(top_idx);
      if (source_layer.layer().type() == "padding") {
        // This layer has a padding layer as input -- check that it is a conv
        // layer or a pooling layer and takes only one input.  Also check that
        // the padding layer input has only one input and one output.  Other
        // cases have undefined behavior in Caffe.
        if( !( (layer_param.type() == "conv") || (layer_param.type() == "pool")) ) {
	  rt_err( "Padding layer input to non-convolutional / non-pooling layer type " + layer_param.type() );
	}
        if( layer_connection.bottom_size() != 1 ) { rt_err( "Conv Layer must take a single blob as input." ); }
        if( source_layer.bottom_size() != 1 ) { rt_err( "Padding Layer must take a single blob as input." ); }
        if( source_layer.top_size() != 1 ) { rt_err( "Padding Layer must produce a single blob as output." ); }
        int layer_index = param_upgraded_pad->layers_size() - 1;
        param_upgraded_pad->mutable_layers(layer_index)->mutable_layer()
            ->set_pad(source_layer.layer().pad());
        param_upgraded_pad->mutable_layers(layer_index)
            ->set_bottom(j, source_layer.bottom(0));
      }
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      blob_name_to_last_top_idx[blob_name] = i;
    }
  }
}

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type) {
  if (type == "accuracy") {
    return V1LayerParameter_LayerType_ACCURACY;
  } else if (type == "bnll") {
    return V1LayerParameter_LayerType_BNLL;
  } else if (type == "concat") {
    return V1LayerParameter_LayerType_CONCAT;
  } else if (type == "conv") {
    return V1LayerParameter_LayerType_CONVOLUTION;
  } else if (type == "data") {
    return V1LayerParameter_LayerType_DATA;
  } else if (type == "dropout") {
    return V1LayerParameter_LayerType_DROPOUT;
  } else if (type == "euclidean_loss") {
    return V1LayerParameter_LayerType_EUCLIDEAN_LOSS;
  } else if (type == "flatten") {
    return V1LayerParameter_LayerType_FLATTEN;
  } else if (type == "hdf5_data") {
    return V1LayerParameter_LayerType_HDF5_DATA;
  } else if (type == "hdf5_output") {
    return V1LayerParameter_LayerType_HDF5_OUTPUT;
  } else if (type == "im2col") {
    return V1LayerParameter_LayerType_IM2COL;
  } else if (type == "images") {
    return V1LayerParameter_LayerType_IMAGE_DATA;
  } else if (type == "infogain_loss") {
    return V1LayerParameter_LayerType_INFOGAIN_LOSS;
  } else if (type == "innerproduct") {
    return V1LayerParameter_LayerType_INNER_PRODUCT;
  } else if (type == "lrn") {
    return V1LayerParameter_LayerType_LRN;
  } else if (type == "multinomial_logistic_loss") {
    return V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS;
  } else if (type == "pool") {
    return V1LayerParameter_LayerType_POOLING;
  } else if (type == "relu") {
    return V1LayerParameter_LayerType_RELU;
  } else if (type == "sigmoid") {
    return V1LayerParameter_LayerType_SIGMOID;
  } else if (type == "softmax") {
    return V1LayerParameter_LayerType_SOFTMAX;
  } else if (type == "softmax_loss") {
    return V1LayerParameter_LayerType_SOFTMAX_LOSS;
  } else if (type == "split") {
    return V1LayerParameter_LayerType_SPLIT;
  } else if (type == "tanh") {
    return V1LayerParameter_LayerType_TANH;
  } else if (type == "window_data") {
    return V1LayerParameter_LayerType_WINDOW_DATA;
  } else {
    rt_err( "Unknown layer name: " + type );
    return V1LayerParameter_LayerType_NONE;
  }
}

bool UpgradeV0LayerParameter(const V1LayerParameter& v0_layer_connection,
                             V1LayerParameter* layer_param) {
  bool is_fully_compatible = true;
  layer_param->Clear();
  for (int i = 0; i < v0_layer_connection.bottom_size(); ++i) {
    layer_param->add_bottom(v0_layer_connection.bottom(i));
  }
  for (int i = 0; i < v0_layer_connection.top_size(); ++i) {
    layer_param->add_top(v0_layer_connection.top(i));
  }
  if (v0_layer_connection.has_layer()) {
    const V0LayerParameter& v0_layer_param = v0_layer_connection.layer();
    if (v0_layer_param.has_name()) {
      layer_param->set_name(v0_layer_param.name());
    }
    const string& type = v0_layer_param.type();
    if (v0_layer_param.has_type()) {
      layer_param->set_type(UpgradeV0LayerType(type));
    }
    for (int i = 0; i < v0_layer_param.blobs_size(); ++i) {
      layer_param->add_blobs()->CopyFrom(v0_layer_param.blobs(i));
    }
    for (int i = 0; i < v0_layer_param.blobs_lr_size(); ++i) {
      layer_param->add_blobs_lr(v0_layer_param.blobs_lr(i));
    }
    for (int i = 0; i < v0_layer_param.weight_decay_size(); ++i) {
      layer_param->add_weight_decay(v0_layer_param.weight_decay(i));
    }
    if (v0_layer_param.has_num_output()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_num_output(
            v0_layer_param.num_output());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_num_output(
            v0_layer_param.num_output());
      } else {
        unk_param_err( "num_output", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_biasterm()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_bias_term(
            v0_layer_param.biasterm());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_bias_term(
            v0_layer_param.biasterm());
      } else {
        unk_param_err( "biasterm", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_weight_filler()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->
            mutable_weight_filler()->CopyFrom(v0_layer_param.weight_filler());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->
            mutable_weight_filler()->CopyFrom(v0_layer_param.weight_filler());
      } else {
        unk_param_err( "weight_filler", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_bias_filler()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->
            mutable_bias_filler()->CopyFrom(v0_layer_param.bias_filler());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->
            mutable_bias_filler()->CopyFrom(v0_layer_param.bias_filler());
      } else {
        unk_param_err( "bias_filler", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_pad()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->add_pad(v0_layer_param.pad());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_pad(v0_layer_param.pad());
      } else {
        unk_param_err( "pad", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_kernelsize()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->add_kernel_size(
            v0_layer_param.kernelsize());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_kernel_size(
            v0_layer_param.kernelsize());
      } else {
        unk_param_err( "kernelsize", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_group()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_group(
            v0_layer_param.group());
      } else {
        unk_param_err( "group", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_stride()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->add_stride(
            v0_layer_param.stride());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_stride(
            v0_layer_param.stride());
      } else {
        unk_param_err( "stride", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_pool()) {
      if (type == "pool") {
        V0LayerParameter_PoolMethod pool = v0_layer_param.pool();
        switch (pool) {
        case V0LayerParameter_PoolMethod_MAX:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_MAX);
          break;
        case V0LayerParameter_PoolMethod_AVE:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_AVE);
          break;
        case V0LayerParameter_PoolMethod_STOCHASTIC:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_STOCHASTIC);
          break;
        default:
          rt_err( "Unknown pool method " + pool );
          is_fully_compatible = false;
        }
      } else {
        unk_param_err( "pool", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_dropout_ratio()) {
      if (type == "dropout") {
        layer_param->mutable_dropout_param()->set_dropout_ratio(
            v0_layer_param.dropout_ratio());
      } else {
        unk_param_err( "dropout_ratio", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_local_size()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_local_size(
            v0_layer_param.local_size());
      } else {
        unk_param_err( "local_size", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_alpha()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_alpha(v0_layer_param.alpha());
      } else {
        unk_param_err( "alpha", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_beta()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_beta(v0_layer_param.beta());
      } else {
        unk_param_err( "beta", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_k()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_k(v0_layer_param.k());
      } else {
        unk_param_err( "k", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_source()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_source(v0_layer_param.source());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "infogain_loss") {
        layer_param->mutable_infogain_loss_param()->set_source(
            v0_layer_param.source());
      } else {
        unk_param_err( "source", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_scale()) {
      layer_param->mutable_transform_param()->
          set_scale(v0_layer_param.scale());
    }
    if (v0_layer_param.has_meanfile()) {
      layer_param->mutable_transform_param()->
          set_mean_file(v0_layer_param.meanfile());
    }
    if (v0_layer_param.has_batchsize()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else {
        unk_param_err( "batchsize", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_cropsize()) {
      layer_param->mutable_transform_param()->
          set_crop_size(v0_layer_param.cropsize());
    }
    if (v0_layer_param.has_mirror()) {
      layer_param->mutable_transform_param()->
          set_mirror(v0_layer_param.mirror());
    }
    if (v0_layer_param.has_rand_skip()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else {
        unk_param_err( "rand_skip", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_shuffle_images()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_shuffle(
            v0_layer_param.shuffle_images());
      } else {
        unk_param_err( "shuffle", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_new_height()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_new_height(
            v0_layer_param.new_height());
      } else {
        unk_param_err( "new_height", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_new_width()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_new_width(
            v0_layer_param.new_width());
      } else {
        unk_param_err( "new_width", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_concat_dim()) {
      if (type == "concat") {
        layer_param->mutable_concat_param()->set_concat_dim(
            v0_layer_param.concat_dim());
      } else {
        unk_param_err( "concat_dim", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_fg_threshold()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_fg_threshold(
            v0_layer_param.det_fg_threshold());
      } else {
        unk_param_err( "det_fg_threshold", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_bg_threshold()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_bg_threshold(
            v0_layer_param.det_bg_threshold());
      } else {
        unk_param_err( "det_bg_threshold", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_fg_fraction()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_fg_fraction(
            v0_layer_param.det_fg_fraction());
      } else {
        unk_param_err( "det_fg_fraction", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_context_pad()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_context_pad(
            v0_layer_param.det_context_pad());
      } else {
        unk_param_err( "det_context_pad", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_crop_mode()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_crop_mode(
            v0_layer_param.det_crop_mode());
      } else {
        unk_param_err( "det_crop_mode", type );
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_hdf5_output_param()) {
      if (type == "hdf5_output") {
        layer_param->mutable_hdf5_output_param()->CopyFrom(
            v0_layer_param.hdf5_output_param());
      } else {
        unk_param_err( "hdf5_output_param", type );
        is_fully_compatible = false;
      }
    }
  }
  return is_fully_compatible;
}


bool UpgradeV0Net(const NetParameter& v0_net_param_padding_layers,
                  NetParameter* net_param) {
  // First upgrade padding layers to padded conv layers.
  NetParameter v0_net_param;
  UpgradeV0PaddingLayers(v0_net_param_padding_layers, &v0_net_param);
  // Now upgrade layer parameters.
  bool is_fully_compatible = true;
  net_param->Clear();
  if (v0_net_param.has_name()) {
    net_param->set_name(v0_net_param.name());
  }
  for (int i = 0; i < v0_net_param.layers_size(); ++i) {
    is_fully_compatible &= UpgradeV0LayerParameter(v0_net_param.layers(i),
                                                   net_param->add_layers());
  }
  for (int i = 0; i < v0_net_param.input_size(); ++i) {
    net_param->add_input(v0_net_param.input(i));
  }
  for (int i = 0; i < v0_net_param.input_dim_size(); ++i) {
    net_param->add_input_dim(v0_net_param.input_dim(i));
  }
  if (v0_net_param.has_force_backward()) {
    net_param->set_force_backward(v0_net_param.force_backward());
  }
  return is_fully_compatible;
}

bool NetNeedsDataUpgrade(const NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).type() == V1LayerParameter_LayerType_DATA) {
      DataParameter layer_param = net_param.layers(i).data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
    if (net_param.layers(i).type() == V1LayerParameter_LayerType_IMAGE_DATA) {
      ImageDataParameter layer_param = net_param.layers(i).image_data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
    if (net_param.layers(i).type() == V1LayerParameter_LayerType_WINDOW_DATA) {
      WindowDataParameter layer_param = net_param.layers(i).window_data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
  }
  return false;
}

#define CONVERT_LAYER_TRANSFORM_PARAM(TYPE, Name, param_name) \
  do { \
    if (net_param->layers(i).type() == V1LayerParameter_LayerType_##TYPE) { \
      Name##Parameter* layer_param = \
          net_param->mutable_layers(i)->mutable_##param_name##_param(); \
      TransformationParameter* transform_param = \
          net_param->mutable_layers(i)->mutable_transform_param(); \
      if (layer_param->has_scale()) { \
        transform_param->set_scale(layer_param->scale()); \
        layer_param->clear_scale(); \
      } \
      if (layer_param->has_mean_file()) { \
        transform_param->set_mean_file(layer_param->mean_file()); \
        layer_param->clear_mean_file(); \
      } \
      if (layer_param->has_crop_size()) { \
        transform_param->set_crop_size(layer_param->crop_size()); \
        layer_param->clear_crop_size(); \
      } \
      if (layer_param->has_mirror()) { \
        transform_param->set_mirror(layer_param->mirror()); \
        layer_param->clear_mirror(); \
      } \
    } \
  } while (0)

void UpgradeNetDataTransformation(NetParameter* net_param) {
  for (int i = 0; i < net_param->layers_size(); ++i) {
    CONVERT_LAYER_TRANSFORM_PARAM(DATA, Data, data);
    CONVERT_LAYER_TRANSFORM_PARAM(IMAGE_DATA, ImageData, image_data);
    CONVERT_LAYER_TRANSFORM_PARAM(WINDOW_DATA, WindowData, window_data);
  }
}

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type) {
  switch (type) {
  case V1LayerParameter_LayerType_NONE:
    return "";
  case V1LayerParameter_LayerType_ABSVAL:
    return "AbsVal";
  case V1LayerParameter_LayerType_ACCURACY:
    return "Accuracy";
  case V1LayerParameter_LayerType_ARGMAX:
    return "ArgMax";
  case V1LayerParameter_LayerType_BNLL:
    return "BNLL";
  case V1LayerParameter_LayerType_CONCAT:
    return "Concat";
  case V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
    return "ContrastiveLoss";
  case V1LayerParameter_LayerType_CONVOLUTION:
    return "Convolution";
  case V1LayerParameter_LayerType_DECONVOLUTION:
    return "Deconvolution";
  case V1LayerParameter_LayerType_DATA:
    return "Data";
  case V1LayerParameter_LayerType_DROPOUT:
    return "Dropout";
  case V1LayerParameter_LayerType_DUMMY_DATA:
    return "DummyData";
  case V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
    return "EuclideanLoss";
  case V1LayerParameter_LayerType_ELTWISE:
    return "Eltwise";
  case V1LayerParameter_LayerType_EXP:
    return "Exp";
  case V1LayerParameter_LayerType_FLATTEN:
    return "Flatten";
  case V1LayerParameter_LayerType_HDF5_DATA:
    return "HDF5Data";
  case V1LayerParameter_LayerType_HDF5_OUTPUT:
    return "HDF5Output";
  case V1LayerParameter_LayerType_HINGE_LOSS:
    return "HingeLoss";
  case V1LayerParameter_LayerType_IM2COL:
    return "Im2col";
  case V1LayerParameter_LayerType_IMAGE_DATA:
    return "ImageData";
  case V1LayerParameter_LayerType_INFOGAIN_LOSS:
    return "InfogainLoss";
  case V1LayerParameter_LayerType_INNER_PRODUCT:
    return "InnerProduct";
  case V1LayerParameter_LayerType_LRN:
    return "LRN";
  case V1LayerParameter_LayerType_MEMORY_DATA:
    return "MemoryData";
  case V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
    return "MultinomialLogisticLoss";
  case V1LayerParameter_LayerType_MVN:
    return "MVN";
  case V1LayerParameter_LayerType_POOLING:
    return "Pooling";
  case V1LayerParameter_LayerType_POWER:
    return "Power";
  case V1LayerParameter_LayerType_RELU:
    return "ReLU";
  case V1LayerParameter_LayerType_SIGMOID:
    return "Sigmoid";
  case V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
    return "SigmoidCrossEntropyLoss";
  case V1LayerParameter_LayerType_SILENCE:
    return "Silence";
  case V1LayerParameter_LayerType_SOFTMAX:
    return "Softmax";
  case V1LayerParameter_LayerType_SOFTMAX_LOSS:
    return "SoftmaxWithLoss";
  case V1LayerParameter_LayerType_SPLIT:
    return "Split";
  case V1LayerParameter_LayerType_SLICE:
    return "Slice";
  case V1LayerParameter_LayerType_TANH:
    return "TanH";
  case V1LayerParameter_LayerType_WINDOW_DATA:
    return "WindowData";
  case V1LayerParameter_LayerType_THRESHOLD:
    return "Threshold";
  default:
    rt_err( "Unknown V1LayerParameter layer type: " + type );
    return "";
  }
}


bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param) {
  layer_param->Clear();
  bool is_fully_compatible = true;
  for (int i = 0; i < v1_layer_param.bottom_size(); ++i) {
    layer_param->add_bottom(v1_layer_param.bottom(i));
  }
  for (int i = 0; i < v1_layer_param.top_size(); ++i) {
    layer_param->add_top(v1_layer_param.top(i));
  }
  if (v1_layer_param.has_name()) {
    layer_param->set_name(v1_layer_param.name());
  }
  for (int i = 0; i < v1_layer_param.include_size(); ++i) {
    layer_param->add_include()->CopyFrom(v1_layer_param.include(i));
  }
  for (int i = 0; i < v1_layer_param.exclude_size(); ++i) {
    layer_param->add_exclude()->CopyFrom(v1_layer_param.exclude(i));
  }
  if (v1_layer_param.has_type()) {
    layer_param->set_type(UpgradeV1LayerType(v1_layer_param.type()));
  }
  for (int i = 0; i < v1_layer_param.blobs_size(); ++i) {
    layer_param->add_blobs()->CopyFrom(v1_layer_param.blobs(i));
  }
  for (int i = 0; i < v1_layer_param.param_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_name(v1_layer_param.param(i));
  }
  ParamSpec_DimCheckMode mode;
  for (int i = 0; i < v1_layer_param.blob_share_mode_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    switch (v1_layer_param.blob_share_mode(i)) {
    case V1LayerParameter_DimCheckMode_STRICT:
      mode = ParamSpec_DimCheckMode_STRICT;
      break;
    case V1LayerParameter_DimCheckMode_PERMISSIVE:
      mode = ParamSpec_DimCheckMode_PERMISSIVE;
      break;
    default:
      rt_err( "Unknown blob_share_mode: " + v1_layer_param.blob_share_mode(i) );
      break;
    }
    layer_param->mutable_param(i)->set_share_mode(mode);
  }
  for (int i = 0; i < v1_layer_param.blobs_lr_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_lr_mult(v1_layer_param.blobs_lr(i));
  }
  for (int i = 0; i < v1_layer_param.weight_decay_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_decay_mult(
        v1_layer_param.weight_decay(i));
  }
  for (int i = 0; i < v1_layer_param.loss_weight_size(); ++i) {
    layer_param->add_loss_weight(v1_layer_param.loss_weight(i));
  }
  if (v1_layer_param.has_accuracy_param()) {
    layer_param->mutable_accuracy_param()->CopyFrom(
        v1_layer_param.accuracy_param());
  }
  if (v1_layer_param.has_argmax_param()) {
    layer_param->mutable_argmax_param()->CopyFrom(
        v1_layer_param.argmax_param());
  }
  if (v1_layer_param.has_concat_param()) {
    layer_param->mutable_concat_param()->CopyFrom(
        v1_layer_param.concat_param());
  }
  if (v1_layer_param.has_contrastive_loss_param()) {
    layer_param->mutable_contrastive_loss_param()->CopyFrom(
        v1_layer_param.contrastive_loss_param());
  }
  if (v1_layer_param.has_convolution_param()) {
    layer_param->mutable_convolution_param()->CopyFrom(
        v1_layer_param.convolution_param());
  }
  if (v1_layer_param.has_data_param()) {
    layer_param->mutable_data_param()->CopyFrom(
        v1_layer_param.data_param());
  }
  if (v1_layer_param.has_dropout_param()) {
    layer_param->mutable_dropout_param()->CopyFrom(
        v1_layer_param.dropout_param());
  }
  if (v1_layer_param.has_dummy_data_param()) {
    layer_param->mutable_dummy_data_param()->CopyFrom(
        v1_layer_param.dummy_data_param());
  }
  if (v1_layer_param.has_eltwise_param()) {
    layer_param->mutable_eltwise_param()->CopyFrom(
        v1_layer_param.eltwise_param());
  }
  if (v1_layer_param.has_exp_param()) {
    layer_param->mutable_exp_param()->CopyFrom(
        v1_layer_param.exp_param());
  }
  if (v1_layer_param.has_hdf5_data_param()) {
    layer_param->mutable_hdf5_data_param()->CopyFrom(
        v1_layer_param.hdf5_data_param());
  }
  if (v1_layer_param.has_hdf5_output_param()) {
    layer_param->mutable_hdf5_output_param()->CopyFrom(
        v1_layer_param.hdf5_output_param());
  }
  if (v1_layer_param.has_hinge_loss_param()) {
    layer_param->mutable_hinge_loss_param()->CopyFrom(
        v1_layer_param.hinge_loss_param());
  }
  if (v1_layer_param.has_image_data_param()) {
    layer_param->mutable_image_data_param()->CopyFrom(
        v1_layer_param.image_data_param());
  }
  if (v1_layer_param.has_infogain_loss_param()) {
    layer_param->mutable_infogain_loss_param()->CopyFrom(
        v1_layer_param.infogain_loss_param());
  }
  if (v1_layer_param.has_inner_product_param()) {
    layer_param->mutable_inner_product_param()->CopyFrom(
        v1_layer_param.inner_product_param());
  }
  if (v1_layer_param.has_lrn_param()) {
    layer_param->mutable_lrn_param()->CopyFrom(
        v1_layer_param.lrn_param());
  }
  if (v1_layer_param.has_memory_data_param()) {
    layer_param->mutable_memory_data_param()->CopyFrom(
        v1_layer_param.memory_data_param());
  }
  if (v1_layer_param.has_mvn_param()) {
    layer_param->mutable_mvn_param()->CopyFrom(
        v1_layer_param.mvn_param());
  }
  if (v1_layer_param.has_pooling_param()) {
    layer_param->mutable_pooling_param()->CopyFrom(
        v1_layer_param.pooling_param());
  }
  if (v1_layer_param.has_power_param()) {
    layer_param->mutable_power_param()->CopyFrom(
        v1_layer_param.power_param());
  }
  if (v1_layer_param.has_relu_param()) {
    layer_param->mutable_relu_param()->CopyFrom(
        v1_layer_param.relu_param());
  }
  if (v1_layer_param.has_sigmoid_param()) {
    layer_param->mutable_sigmoid_param()->CopyFrom(
        v1_layer_param.sigmoid_param());
  }
  if (v1_layer_param.has_softmax_param()) {
    layer_param->mutable_softmax_param()->CopyFrom(
        v1_layer_param.softmax_param());
  }
  if (v1_layer_param.has_slice_param()) {
    layer_param->mutable_slice_param()->CopyFrom(
        v1_layer_param.slice_param());
  }
  if (v1_layer_param.has_tanh_param()) {
    layer_param->mutable_tanh_param()->CopyFrom(
        v1_layer_param.tanh_param());
  }
  if (v1_layer_param.has_threshold_param()) {
    layer_param->mutable_threshold_param()->CopyFrom(
        v1_layer_param.threshold_param());
  }
  if (v1_layer_param.has_window_data_param()) {
    layer_param->mutable_window_data_param()->CopyFrom(
        v1_layer_param.window_data_param());
  }
  if (v1_layer_param.has_transform_param()) {
    layer_param->mutable_transform_param()->CopyFrom(
        v1_layer_param.transform_param());
  }
  if (v1_layer_param.has_loss_param()) {
    layer_param->mutable_loss_param()->CopyFrom(
        v1_layer_param.loss_param());
  }
  if (v1_layer_param.has_layer()) {
    rt_err( "Input NetParameter to upgrade from V1->V2 has V0 layer -- aborting." );
    is_fully_compatible = false;
  }
  return is_fully_compatible;
}


bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param) {
  bool is_fully_compatible = true;
  if (v1_net_param.layer_size() > 0) {
    rt_err( "Input NetParameter to be upgraded from V1->V2 already specifies 'layer' "
	    "fields; these would be ignored in an upgrade. aborting." );
    is_fully_compatible = false;
  }
  net_param->CopyFrom(v1_net_param);
  net_param->clear_layers();
  net_param->clear_layer();
  for (int i = 0; i < v1_net_param.layers_size(); ++i) {
    if (!UpgradeV1LayerParameter(v1_net_param.layers(i),
                                 net_param->add_layer())) {
      rt_err( "Upgrade from V1->v2 of input layer " + str(i) + " failed." );
      is_fully_compatible = false;
    }
  }
  return is_fully_compatible;
}

bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param) {
  bool success = true;
  if (NetNeedsV0ToV1Upgrade(*param)) {
    rt_err( "V0 upgrade no longer supported by boda/caffe" );
#if 0
    // NetParameter was specified using the old style (V0LayerParameter); try to
    // upgrade it.
    LOG(ERROR) << "Attempting to upgrade input file specified using deprecated "
               << "V0LayerParameter: " << param_file;
    NetParameter original_param(*param);
    if (!UpgradeV0Net(original_param, param)) {
      success = false;
      LOG(ERROR) << "Warning: had one or more problems upgrading "
          << "V0NetParameter to NetParameter (see above); continuing anyway.";
    } else {
      LOG(INFO) << "Successfully upgraded file specified using deprecated "
                << "V0LayerParameter";
    }
    LOG(ERROR) << "Note that future Caffe releases will not support "
        << "V0NetParameter; use ./build/tools/upgrade_net_proto_text for "
        << "prototxt and ./build/tools/upgrade_net_proto_binary for model "
        << "weights upgrade this and any other net protos to the new format.";
#endif
  }
  // NetParameter uses old style data transformation fields; try to upgrade it.
  if (NetNeedsDataUpgrade(*param)) {
    rt_err( "data param upgrade not supported by boda/caffe" );
#if 0
    LOG(ERROR) << "Attempting to upgrade input file specified using deprecated "
               << "transformation parameters: " << param_file;
    UpgradeNetDataTransformation(param);
    LOG(INFO) << "Successfully upgraded file specified using deprecated "
              << "data transformation parameters.";
    LOG(ERROR) << "Note that future Caffe releases will only support "
               << "transform_param messages for transformation fields.";
#endif
  }
  if (NetNeedsV1ToV2Upgrade(*param)) {
    // too noisy ... but maybe some warning should be preserved somewhere?
    //LOG(ERROR) << "Attempting to upgrade input file specified using deprecated " << "V1LayerParameter: " << param_file;
    NetParameter original_param(*param);
    if (!UpgradeV1Net(original_param, param)) {
      rt_err( "V1->V2 prototxt upgrade: one or more upgrade problems, aborting." ); // stricter than caffe
    }
#if 0
    if (!UpgradeV1Net(original_param, param)) {
      success = false;
      LOG(ERROR) << "Warning: had one or more problems upgrading "
          << "V1LayerParameter (see above); continuing anyway.";
    } else {

      LOG(INFO) << "Successfully upgraded file specified using deprecated "
                << "V1LayerParameter";
    }
#endif
  }
  return success;
}

// mwm: modified versions of Net::StateMeetsRule() and Net::FilterNet() from net.cpp:

  bool StateMeetsRule( const NetState & state, const NetStateRule & rule ) {
    // Check whether the rule is broken due to phase.
    if(rule.has_phase()) { if (rule.phase() != state.phase()) { return false; } }
    // Check whether the rule is broken due to min level.
    if (rule.has_min_level()) { if (state.level() < rule.min_level()) { return false; } }
    // Check whether the rule is broken due to max level.
    if (rule.has_max_level()) { if (state.level() > rule.max_level()) { return false; } }
    // Check whether the rule is broken due to stage. The NetState must
    // contain ALL of the rule's stages to meet it.
    for (int i = 0; i < rule.stage_size(); ++i) {
      // Check that the NetState contains the rule's ith stage.
      bool has_stage = false;
      for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
	if (rule.stage(i) == state.stage(j)) { has_stage = true; }
      }
      if (!has_stage) { return false; }
    }
    // Check whether the rule is broken due to not_stage. The NetState must
    // contain NONE of the rule's not_stages to meet it.
    for (int i = 0; i < rule.not_stage_size(); ++i) {
      // Check that the NetState contains the rule's ith not_stage.
      bool has_stage = false;
      for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
	if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
      }
      if (has_stage) { return false; }
    }
    return true;
  }
  bool layer_included_for_state( LayerParameter const & layer_param, NetState const & net_state ) {   // modified from inner loop of from FilterNet()
    if( layer_param.include_size() && layer_param.exclude_size() ) { rt_err("Specify either include rules or exclude rules; not both."); }
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j))) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j))) {
        layer_included = true;
      }
    }
    return layer_included;
  }

}  // namespace boda_caffe
