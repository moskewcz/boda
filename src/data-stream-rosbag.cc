// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"


#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

namespace boda 
{


  typedef shared_ptr< rosbag::View > p_rosbag_view;
  
  struct data_stream_rosbag_t : virtual public nesi, public data_stream_t // NESI(help="parse mxnet-brick-style-serialized data stream into data blocks",
                                     // bases=["data_stream_t"], type_id="rosbag-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t fn; //NESI(req=1,help="input filename")
    vect_string topics; //NESI(req=1,help="list of topics to read from bag")
    
    rosbag::Bag bag;
    p_rosbag_view view;
    rosbag::View::iterator vi;
    
    virtual string get_pos_info_str( void ) { return strprintf( "rosbag: status <TODO>" ); }

    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( vi == view->end() ) { return ret; }
      ret.subblocks = make_shared<vect_data_block_t>(topics.size());
      for( uint32_t i = 0; i != topics.size(); ++i ) {
        rosbag::MessageInstance const & msg = *vi;
        printf( "msg.getTopic()=%s\n", str(msg.getTopic()).c_str() );
        //ret.subblocks->at(i) = ;
      }
      ++vi;
      return ret;
    }
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      bag.open( fn.exp, rosbag::bagmode::Read );
      view = make_shared< rosbag::View >( bag, rosbag::TopicQuery(topics) );
      vi = view->begin();
    }
  };
  
#include"gen/data-stream-rosbag.cc.nesi_gen.cc"

}
