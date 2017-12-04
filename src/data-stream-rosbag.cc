// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"timers.H"
#include"img_io.H"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// #include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
// #include <sensor_msgs/CameraInfo.h>

// for transforms
#include <tf2/convert.h>
#include <tf2/buffer_core.h>
#include <tf2_msgs/TFMessage.h>
#include <tf/tfMessage.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace boda 
{
  // FIXME: move to data-stream.H
  p_data_block_t make_nda_db( string const & tag, p_nda_t const & nda ) {
    p_data_block_t ret = make_shared< data_block_t >();
    ret->tag = tag;
    ret->nda = nda;
    return ret;
  }
  
  using sensor_msgs::PointField;
  //note: no need to use a vect of PointField directly; we just use the one in our temp PointCloud2
  //typedef vector< PointField > vect_PointField;
  using sensor_msgs::PointCloud2;

  typedef shared_ptr< rosbag::View > p_rosbag_view;
  typedef rosbag::MessageInstance message_instance_t;
  typedef std::deque< message_instance_t > deque_message_instance_t;
  typedef vector< deque_message_instance_t > vect_deque_message_instance_t;

  using geometry_msgs::Point;
  typedef vector< Point > vect_ros_point_t;

  using visualization_msgs::Marker;
  typedef vector< Marker > vect_ros_marker_t;

  typedef map< string, vect_ros_marker_t > map_str_vect_ros_marker_t;
  
  uint64_t get_ros_timestamp( ros::Duration const & t ) { return secs_and_nsecs_to_nsecs_signed( t.sec, t.nsec ); }
  uint64_t get_ros_timestamp( ros::Time const & t ) { return secs_and_nsecs_to_nsecs_signed( t.sec, t.nsec ); }
  uint64_t get_ros_msg_timestamp( message_instance_t const & msg ) { return get_ros_timestamp( msg.getTime() ); }
  uint64_t ts_delta( uint64_t const & a, uint64_t const & b ) { return ( a > b ) ? ( a - b ) : ( b - a ); }
  
  struct pc2_gf_t {
    string name;
    uint32_t offset;
    pc2_gf_t( string const & name_ ) : name(name_), offset(uint32_t_const_max) { }
  };
  typedef vector< pc2_gf_t > vect_pc2_gf_t; 

  // if there are multiple topics, the first topic will be considered the 'primary' topic, and one data block will be
  // emitted per message on that topic. other topics will be synced to the primary topic by choosing the message from
  // each other topic closest in time to the primary topic message. in general, this means that messages on non-primary
  // topics can be dropped or stuttered. note that for some types of messages, however, multiple messages may instead be
  // merged together, such that some set of non-primary topic messages (all near in time to the primary message) are
  // emitted. in particular, one case is that, for some secondary topics, all messages will be emitted exactly once,
  // attached to the primary topic message that they are closest to.
  struct data_stream_rosbag_src_t : virtual public nesi, public data_stream_t // NESI(help="parse mxnet-brick-style-serialized data stream into data blocks",
                                    // bases=["data_stream_t"], type_id="rosbag-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t fn; //NESI(req=1,help="input filename")
    vect_string topics; //NESI(req=1,help="list of topics to read from bag")
    string frame_id; //NESI(default="base_link",help="for output points, what frame id to transform into")

    rosbag::Bag bag;
    p_rosbag_view view;
    rosbag::View::iterator vi;

    shared_ptr< tf2::BufferCore > tf2_bc;
    
    vect_deque_message_instance_t msg_deques; // deques for storing non-primary messages
    map_str_uint32_t topic_to_ix;

    vect_pc2_gf_t pc2_gfs; // named fields we must be able to extract to parse PointCloud2 messages (see init)

    // temporaries used to store into-frame_id-transformed messages
    sensor_msgs::PointCloud2 pc2_tf_buf; 
    geometry_msgs::PoseStamped pose_tf_buf_in;
    geometry_msgs::PoseStamped pose_tf_buf_out;

    // cached/buffered secondary topic data (stored per-topic)
    map_str_vect_ros_marker_t topic_marker_bufs;
    
    virtual string get_pos_info_str( void ) { return strprintf( "rosbag: status <TODO>" ); }

    // returns false if we get to end-of-stream before finding a primary msg, otherwise returns true
    bool read_up_to_primary_topic_msg( void ) {
      // read messages until we get to a primary topic message.
      while( 1 ) {
        if( vi == view->end() ) { return false; }
        rosbag::MessageInstance const & msg = *vi;
        if( msg.getTopic() == topics[0] ) { return true; }
        if( msg.getTopic() == "/tf" ) {
          if( msg.getDataType() == "geometry_msgs/TransformStamped" ) {
            geometry_msgs::TransformStamped::ConstPtr tfs = msg.instantiate<geometry_msgs::TransformStamped>();
            tf2_bc->setTransform( *tfs, string("bag"), false );
          } else if( msg.getDataType() == "tf2_msgs/TFMessage" ) {
            tf2_msgs::TFMessage::ConstPtr tfss = msg.instantiate<tf2_msgs::TFMessage>();
            for( auto tfs = tfss->transforms.begin(); tfs != tfss->transforms.end(); ++tfs ) {
              tf2_bc->setTransform( *tfs, string("bag"), false );
            }
          } else if( msg.getDataType() == "tf/tfMessage" ) {
            tf::tfMessage::ConstPtr tfss = msg.instantiate<tf::tfMessage>();
            for( auto tfs = tfss->transforms.begin(); tfs != tfss->transforms.end(); ++tfs ) {
              tf2_bc->setTransform( *tfs, string("bag"), false );
            }
          } else {
            rt_err( strprintf( "unhandled message type in /tf topic with msg.getDataType()=%s\n", str(msg.getDataType()).c_str() ) );
          }
        } else {
          // otherwise process/store secondary topic msg
          uint32_t topic_ix = must_find( topic_to_ix, msg.getTopic() );
          msg_deques.at( topic_ix ).push_back( msg );
        }
        ++vi;
      }
    }

    // if prim_msg is null, msg is a primary message. otherise msg is secondary, and prim_msg is the corresponding primary msg.
    // if is_stale is true, don't emit a data_block, but use a topic/type dependent method to store/buffer any desired
    // information about msg to include in the next block for the given topic. note that, for correctness any cached
    // into must be stored *per topic*, using a map or the like. currently, most topics/types just drop stale
    // messages. but some, like markers, cache/buffer/batch messages.
    void msg_to_db( message_instance_t const * const prim_msg, bool const & is_stale,
                    data_block_t & ret, message_instance_t const & msg ) {
      //printf( "msg.getDataType()=%s\n", str(msg.getDataType()).c_str() );
      if( !prim_msg ) { assert_st( !is_stale ); } // should emit every primary msg, so it should never be marked stale
      if( msg.getDataType() == "sensor_msgs/Image" ) {
        if( is_stale ) { return; } // stale policy: drop
        sensor_msgs::Image::ConstPtr img = msg.instantiate<sensor_msgs::Image>();
        //printf( "img->height=%s img->width=%s img->encoding=%s\n", str(img->height).c_str(), str(img->width).c_str(), str(img->encoding).c_str() );
        if( img->encoding != "bayer_bggr8" ) { rt_err( "unsupported image encoding in rosbag: " + img->encoding ); }
        
        assert_st( (img->height * img->step) == img->data.size() );
        p_nda_uint8_t img_nda = make_shared<nda_uint8_t>( dims_t{ vect_uint32_t{uint32_t(img->height), uint32_t(img->width)}, vect_string{ "y","x" },"uint8_t" }); // note: for now, always in in bggr format ...
        // copy image data to packed nda. FIXME: if we had nda padding, we could borrow, or at least copy in one
        // block. also if we checked for the un-padded case, we could do similar for that case at least.
        for( uint32_t y = 0; y != img->height; ++y ) { 
          uint8_t const * rb = &img->data[img->step*y];
          std::copy( rb, rb + img->width, &img_nda->at1(y) );
        }
        ret.nda = img_nda;
        ret.meta = "image";
        //uint64_t const img_ts = secs_and_nsecs_to_nsecs_signed( img->header.stamp.sec, img->header.stamp.nsec );
      } else if( msg.getDataType() == "sensor_msgs/PointCloud2" ) {
        if( is_stale ) { return; } // stale policy: drop
        sensor_msgs::PointCloud2::ConstPtr pc2 = msg.instantiate<sensor_msgs::PointCloud2>();
        uint32_t gfs_found = 0;
        for( auto i = pc2->fields.begin(); i != pc2->fields.end(); ++i ) {
          for( auto gf = pc2_gfs.begin(); gf != pc2_gfs.end(); ++gf ) {
            if( i->name == gf->name ) {
              if( i->datatype != 7 ) { rt_err( "unsupported PointField datatype: " + str(i->datatype) ); }
              if( i->count != 1 ) { rt_err( "unsupported not-equal-to-one PointField count: " + str(i->count) ); }
              gf->offset = i->offset;
              ++gfs_found;
            }
          }
        }
        if( gfs_found != pc2_gfs.size() ) {
          rt_err( strprintf( "can't parse PointCloud2, only found gfs_found=%s fields, but needed pc2_gfs.size()=%s fields.", str(gfs_found).c_str(), str(pc2_gfs.size()).c_str() ) );
        }
        geometry_msgs::TransformStamped tf = tf2_bc->lookupTransform( frame_id, pc2->header.frame_id, ros::Time(0) ); // note: use last-known transform, not 'correct' one, to avoid 'extrapolate into past' error. was: // pc2->header.stamp );
        tf2::doTransform( *pc2, pc2_tf_buf, tf );
        p_nda_float_t pc2_nda = make_shared<nda_float_t>( dims_t{ vect_uint32_t{uint32_t(pc2_tf_buf.height), uint32_t(pc2_tf_buf.width), uint32_t(pc2_gfs.size())}, vect_string{ "y","x","p" },"float" });
        for( uint32_t y = 0; y != pc2_tf_buf.height; ++y ) {
          uint8_t const * row = &pc2_tf_buf.data[pc2_tf_buf.row_step*y];
          for( uint32_t x = 0; x != pc2_tf_buf.width; ++x ) {
            float * out = &pc2_nda->at2(y,x);
            for( auto gf = pc2_gfs.begin(); gf != pc2_gfs.end(); ++gf ) {
              *out = *((float const *)(row + gf->offset));
              ++out;
            }
            row += pc2_tf_buf.point_step;
          }
        }
        ret.nda = pc2_nda;
        ret.meta = "pointcloud";
      } else if( msg.getDataType() == "visualization_msgs/Marker" ) {
        visualization_msgs::Marker marker = *msg.instantiate<visualization_msgs::Marker>();
        auto & marker_buf = topic_marker_bufs[msg.getTopic()];
        // for now, capture only the position, and only for cube markers
        if( marker.type == 1 ) {
          pose_tf_buf_in.pose = marker.pose; // use a PoseStamped because ... there's no tf2 doTransform for regular Pose or Marker?
          geometry_msgs::TransformStamped tf = tf2_bc->lookupTransform( frame_id, marker.header.frame_id, ros::Time(0) ); // note: use last-known transform, not 'correct' one, to avoid 'extrapolate into past' error. was: // marker->header.stamp );
          tf2::doTransform( pose_tf_buf_in, pose_tf_buf_out, tf );
          marker.pose = pose_tf_buf_out.pose;
          marker_buf.push_back( marker );          
          auto const & pos = marker.pose.position;
          printf( "pos.x=%s pos.y=%s pos.z=%s\n", str(pos.x).c_str(), str(pos.y).c_str(), str(pos.z).c_str() );
        }
        if( is_stale ) { return; } // stale policy: buffer // FIXME: use duration
        // if not stale, get all current points, but into nda.
        // filter buffered markers
        auto o = marker_buf.begin();
        uint64_t cur_time = get_ros_msg_timestamp( prim_msg ? *prim_msg : msg );
        for( uint32_t i = 0; i != marker_buf.size(); ++i ) {
          auto const & m = marker_buf[i];
          // FIXME: not really right i guess, should use original message bag timestamp, or maybe use header stamps for
          // everything? wish i understood the ROS bag msg timestamp vs. header timestamp issue better.
          uint64_t const marker_ts = get_ros_timestamp( m.header.stamp ); 
          //bool const keep = ts_delta( cur_time, marker_ts ) < (1000U*1000U*1000U*10U); // FIXME: use real duration when viz ready
          bool const keep = ts_delta( cur_time, marker_ts ) < get_ros_timestamp( m.lifetime ); // FIXME: real duration version
          if( keep ) { *o = m; o++; }
        }
        marker_buf.erase( o, marker_buf.end() );
        
        // convert buffered markers to pointcloud
        p_nda_float_t marker_nda = make_shared<nda_float_t>( dims_t{ vect_uint32_t{uint32_t(marker_buf.size()), 3}, vect_string{ "v","p" },"float" });
        for( uint32_t i = 0; i != marker_buf.size(); ++i ) {
          auto const & m = marker_buf[i];
          auto const & pt = m.pose.position;
          float * out = &marker_nda->at1(i);
          out[0] = pt.x;
          out[1] = pt.y;
          out[2] = pt.z;
        }
        
        ret.nda = marker_nda;
        ret.meta = "pointcloud";
        ret.set_sdb( make_nda_db( "pt_sz", make_scalar_nda<float>(20.0) ) );
      } else {
        rt_err( "rosbag-src: unhandled ros message type: " + msg.getDataType() );
      }
      ret.tag = "rosbag:"+msg.getTopic();
      ret.timestamp_ns = get_ros_msg_timestamp( msg );
      //printf( "ret.timestamp_ns=%s img_ts=%s\n", str(ret.timestamp_ns).c_str(), str(img_ts).c_str() );
    }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( vi == view->end() ) { return ret; }
      ret.subblocks = make_shared<vect_data_block_t>(topics.size());
      // we should be at a primary topic message, so consume it
      rosbag::MessageInstance const prim_msg = *vi;
      ++vi;
      assert_st( prim_msg.getTopic() == topics[0] );
      read_up_to_primary_topic_msg(); // read up to *next* primary topic msg (if there is one)

      data_block_t pdb = db;
      msg_to_db( 0, 0, pdb, prim_msg );
      ret.subblocks->at(0) = pdb;
      ret.timestamp_ns = ret.subblocks->at(0).timestamp_ns; // use primary timestamp as timeframe timestamp
      
      for( uint32_t i = 1; i != topics.size(); ++i ) {
        data_block_t sdb = db;
        deque_message_instance_t & msg_deque = msg_deques.at(i);
        // first, drop any too-early-to-be-usefull message // FIXME: keep these messages for streams that can accumluate
        while( (msg_deque.size() > 1) && ( get_ros_msg_timestamp( msg_deque[1] ) < ret.timestamp_ns ) ) {
          msg_to_db( &prim_msg, 1, sdb, msg_deque.front() );
          msg_deque.pop_front();
        }
        // now: if there is a second message: (1) it must be >= the primary timestamp, and (2) the prior message must
        // have a timestamp <= the second message (as is always true, assuming message timestamps are monotonic).
        // here, we determine if the first or second message is closer to the primary message.
        bool second_msg_is_closer = (msg_deque.size() > 1) &&
          ( ts_delta( ret.timestamp_ns, get_ros_msg_timestamp( msg_deque[1] ) ) <
            ts_delta( ret.timestamp_ns, get_ros_msg_timestamp( msg_deque[0] ) ) );
        if( second_msg_is_closer ) {
          msg_to_db( &prim_msg, 1, sdb, msg_deque.front() );
          msg_deque.pop_front();
        } // FIXME: as above, accumulate this msg if desired
        
        if( !msg_deque.empty() ) { msg_to_db( &prim_msg, 0, sdb, msg_deque.front() ); }
        ret.subblocks->at(i) = sdb;
      }
      return ret;
    }
    
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      if( topics.empty() ) { rt_err( "rosbag-src: must specify at least one topic (first will be primary)" ); }
      pc2_gfs = { pc2_gf_t{"x"}, pc2_gf_t{"y"}, pc2_gf_t{"z"} };
      tf2_bc = make_shared< tf2::BufferCore >();
      bag.open( fn.exp, rosbag::bagmode::Read );
      vect_string topics_with_tf = topics;
      topics_with_tf.push_back( "/tf" );
      view = make_shared< rosbag::View >( bag, rosbag::TopicQuery(topics_with_tf) );
      vi = view->begin();
      for( uint32_t i = 0; i != topics.size(); ++i ) { must_insert( topic_to_ix, topics[i], i ); }
      msg_deques.resize( topics.size() );
      read_up_to_primary_topic_msg();
    }
  };

  struct pc2_point {
    float x,y,z,intensity;
    uint16_t ring;
  };

  float const meters_to_feet = 3.28084;
  
  struct data_stream_rosbag_sink_t : virtual public nesi, public data_stream_t // NESI(help="parse mxnet-brick-style-serialized data stream into data blocks",
                                     // bases=["data_stream_t"], type_id="rosbag-sink")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t fn; //NESI(req=1,help="output filename")
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    uint32_t append_mode; //NESI(default="0",help="if 1, open bag for append. otherwise, open for writing.")
    uint32_t rot_90; //NESI(default="0",help="if 1, rotate data 90 CW (x_ros=y_in; y_ros=-x_in).")
    uint32_t compress_images_as_jpeg; //NESI(default="0",help="if 1, compress images as jpeg. FIXME/NOTE: can't seem to view the compressed images in rviz, so maybe not such a usefull option currently.")
    float scale_xy; //NESI(default="1.0",help="scale xy points by this value")

    string frame_id; //NESI(default="base_link",help="for output msg headers, what frame id to use")

    vect_string topics; //NESI(req=1,help="list of topics to write to bag, one per sub-block. to omit a topic, use an empty name.")
    rosbag::Bag bag;

    std_msgs::Header msg_header; // used as template/temporary.
    PointCloud2 pc2; // used as template/temporary.
    
    virtual string get_pos_info_str( void ) { return strprintf( "rosbag: status <TODO>" ); }

    virtual data_block_t proc_block( data_block_t const & db ) {
      if( topics.size() != db.num_subblocks() ) {
        rt_err( strprintf( "topics.size()=%s must equal db.num_subblocks()=%s\n",
                           str(topics.size()).c_str(), str(db.num_subblocks()).c_str() ) );
      }
      if( verbose ) { printf( "rosbag-sink: db.info_str()=%s\n", db.info_str().c_str() ); }
      
      assert_st( db.has_subblocks() );
      for( uint32_t i = 0; i != db.subblocks->size(); ++i ) { write_db_to_bag( db.subblocks->at(i), topics.at(i) ); }
      return db;
    }
    void write_db_to_bag( data_block_t const & db, string const & topic ) {
      if( topic.empty() ) { return; } // skip if directed to do so
      ros::Time ros_ts;
      ros_ts.fromNSec( db.timestamp_ns ); // note: we'll use this for both the 'recv' and 'header' timestamp for our gen'd message
      msg_header.seq = db.frame_ix;
      msg_header.stamp = ros_ts;
      msg_header.frame_id = frame_id;

      if( 0 ) { }
      else if( startswith( db.meta, "image" ) || startswith( db.meta, "IMAGEDATA" ) ) {
        if( !db.as_img ) { rt_err( "rosbag-sink: image: expected as_img to be non-null" ); }
        if( compress_images_as_jpeg ) {
          sensor_msgs::CompressedImagePtr img = boost::make_shared< sensor_msgs::CompressedImage >();
          img->header = msg_header;
          img->format = "jpeg";
          p_uint8_with_sz_t img_jpeg = db.as_img->to_jpeg();
          img->data.resize( img_jpeg.sz );
          std::copy( img_jpeg.get(), img_jpeg.get() + img_jpeg.sz, &img->data[0] );
          bag.write( topic + "/compressed", ros_ts, img );
        } else {
          sensor_msgs::ImagePtr img = boost::make_shared< sensor_msgs::Image >();
          img->header = msg_header;
          img->width = db.as_img->sz.d[0];
          img->height = db.as_img->sz.d[1];
          img->encoding = sensor_msgs::image_encodings::RGBA8;
          img->step = db.as_img->row_pitch;
          assert_st( (img->height * img->step) == db.as_img->sz_raw_bytes() );
          img->data.resize( db.as_img->sz_raw_bytes() );
          std::copy( db.as_img->pels.get(), db.as_img->pels.get() + db.as_img->sz_raw_bytes(), &img->data[0] );
          bag.write( topic, ros_ts, img );
        }
      } else if( startswith( db.meta, "pointcloud" ) ) {
        //sensor_msgs::PointCloud2Ptr pc2 = boost::make_shared< sensor_msgs::PointCloud2 >();
        pc2.header = msg_header;

        // p_nda_float_t xyz_nda = make_shared<nda_float_t>( dims_t{ dims_t{ { xy_sz.d[1], xy_sz.d[0], 3 }, {"y","x","xyz"}, "float" }} );
        assert_st( db.nda );
        p_nda_float_t xyz_nda = make_shared< nda_float_t >( db.nda );
        u32_pt_t xy_sz = get_xy_dims( xyz_nda->dims );

        pc2.width = xy_sz.d[0];
        pc2.height = xy_sz.d[1];
        pc2.row_step = pc2.point_step * pc2.width;
        pc2.data.resize( pc2.row_step * pc2.height, 0 );

        pc2_point pt;
        uint8_t * out = &pc2.data[0];
        for( uint32_t y = 0; y != xy_sz.d[1] ; ++y ) {
          for( uint32_t x = 0; x != xy_sz.d[0] ; ++x ) {
            float * const xyz = &xyz_nda->at2(y,x);
            pt.x = xyz[0]; pt.y = xyz[1]; pt.z = xyz[2]; pt.intensity = 50; pt.ring = y;
            if( rot_90 ) { std::swap(pt.x,pt.y); pt.y = -pt.y; }
            if( 1 ) { pt.x *= scale_xy; pt.y *= scale_xy; }
            std::copy( (uint8_t const *)&pt, (uint8_t const *)&pt + sizeof(pt), out );
            out += pc2.point_step;
          }
        }

        bag.write( topic, ros_ts, pc2 );

        // NOTE: ros axis conventions:
        // in relation to body (i.e. 'body_link'); colors in ()s are as in rviz TF view (best guess currently)
        //   x forward (red)
        //   y left (green)
        //   z up (blue)

#if 0
        // FIXME/NOTE: this untested/unused code is from an SO post about writing transforms to a bag, and seems
        // plausible. if we want to use our own frame (i.e. the camera frame), this might be one way to do it.
        // see: https://answers.ros.org/question/65556/write-a-tfmessage-to-bag-file/
        geometry_msgs::TransformStamped msg;
        msg.header = msg_header; 
        msg.child_frame_id = msg_header.frame_id;
        msg.header.frame_id = base_frame_id; // FIXME: set base frame to proper frame here
        msg.transform.translation.x = ???; // FIXME: set xform properly
        tf::tfMessage message;
        message.transforms.push_back( msg );
        bag.write("tf", ros_ts, message); // MWM FIXME: use leading slash for tf topic, i.e. '/tf'?
#endif
        
      } else { rt_err( "rosbag-sink: unhandled db with meta=" + db.meta ); }
    }      
    
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      bag.open( fn.exp, append_mode ? rosbag::bagmode::Append : rosbag::bagmode::Write );
      uint32_t offset = 0;
      vect_string float_fields = {"x","y","z","intensity"}; // note: float type = 7
      PointField pf;
      pf.count = 1;
      pf.datatype = 7;
      for( vect_string::const_iterator i = float_fields.begin(); i != float_fields.end(); ++i ) {
        pf.name = *i;
        pf.offset = offset;
        pc2.fields.push_back( pf );
        offset += sizeof(float);
      }
      // ring, type: uint16_t=4
      pf.name = "ring";
      pf.datatype = 4;
      pf.offset = offset;
      offset += sizeof( uint16_t );
      offset += 2; // FIXME: should we be adding this padding?
      pc2.fields.push_back( pf );

      pc2.is_bigendian = 0;
      pc2.point_step = offset;
      // pc2.row_step --> set per-msg to pc2.point_step * width (although probably constant across msgs)
      // pc2.data --> set per-msg 
      pc2.is_dense = 1;
      
    }
  };
  
#include"gen/data-stream-rosbag.cc.nesi_gen.cc"

}
