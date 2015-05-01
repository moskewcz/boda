// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"lmdbif.H"
#include"caffepb.H"
#include"has_main.H"

#include"disp_util.H" // only display_lmdb_t
#include"asio_util.H" // only display_lmdb_t

#include"caffeif.H" // only test_lmdb_t

namespace boda 
{

  struct lmdb_parse_datums_t : virtual public nesi, public has_main_t // NESI(help="parse caffe-style datums stored in an lmdb",
			       // bases=["has_main_t"], type_id="lmdb_parse_datums")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t db_fn; //NESI(default="%(datasets_dir)/imagenet_classification/ilsvrc12_val_lmdb",help="input lmdb dir filename")
    uint64_t num_to_read; //NESI(default=10,help="read this many records")

    lmdb_state_t lmdb;
    uint64_t tot_num_read; // num read so far

    p_datum_t read_next_datum( void ) {
      assert_st( tot_num_read <= num_to_read );
      if( tot_num_read == num_to_read ) { return p_datum_t(); }
      MDB_val key, data;      
      bool const ret = lmdb.cursor_next( &key, &data );
      if( !ret ) { return p_datum_t(); }
      ++tot_num_read;
      p_datum_t datum = parse_datum( data.mv_data, data.mv_size );
      return datum;
    }

    void read_batch_of_datums( p_nda_float_t & in_batch, vect_uint32_t & labels ) {
      assert_st( labels.empty() );
      MDB_val key, data;   
      assert_st( in_batch->dims.sz() == 4 );
      for( uint32_t i = 0; i != in_batch->dims.dims(0); ++i ) {
	if( tot_num_read == num_to_read ) { return; }
	bool const ret = lmdb.cursor_next( &key, &data );
	if( !ret ) { break; }
	labels.push_back( parse_datum_into( in_batch, i, data.mv_data, data.mv_size ) );
	++tot_num_read;
      }
    }

    void lmdb_open_and_start_read_pass( void ) {
      lmdb.env_open( db_fn.exp, MDB_RDONLY ); 
      lmdb.txn_begin( MDB_RDONLY );
      lmdb.cursor_open();
      tot_num_read = 0;
    }
    
    void main( nesi_init_arg_t * nia ) { 
      lmdb_open_and_start_read_pass();
      while( read_next_datum() ) { }
    }

  };

  struct test_lmdb_t : virtual public nesi, public lmdb_parse_datums_t // NESI(
			 // help="test lmdb with run_cnet_t",
			 // bases=["lmdb_parse_datums_t"], type_id="test_lmdb")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(models_dir)/alexnet/deploy.prototxt,
    //trained_fn=%(models_dir)/alexnet/best.caffemodel,out_layer_name=prob,in_num_imgs=50)",help="cnet running options")

    void main( nesi_init_arg_t * nia ) { 
      run_cnet->setup_cnet(); 
      lmdb_open_and_start_read_pass();
      vect_uint32_t batch_labels_gt;
      vect_uint32_t batch_labels_out;
      uint64_t num_test = 0;
      uint64_t num_pos = 0;
      while( 1 ) {
	batch_labels_gt.clear();
	batch_labels_out.clear();
	read_batch_of_datums( run_cnet->in_batch, batch_labels_gt );
	if( batch_labels_gt.empty() ) { break; } // quit if we run out of data early
	p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
	assert( out_batch->dims.sz() == 4 );
	assert( out_batch->dims.dims(0) >= batch_labels_gt.size() );
	assert( out_batch->dims.dims(1) == 1000 );
	assert( out_batch->dims.dims(2) == 1 );
	assert( out_batch->dims.dims(3) == 1 );
	for( uint32_t i = 0; i != batch_labels_gt.size(); ++i ) {
	  uint32_t max_chan_ix = uint32_t_const_max;
	  float max_chan_val = 0;
	  for( uint32_t j = 0; j != out_batch->dims.dims(1); ++j ) {
	    float const & v = out_batch->at4(i,j,0,0);
	    if( (max_chan_ix == uint32_t_const_max) || (v > max_chan_val) ) { max_chan_ix = j; max_chan_val = v; }
	  }
	  batch_labels_out.push_back( max_chan_ix );
	  ++num_test;
	  if( batch_labels_gt[i] == batch_labels_out[i] ) { ++num_pos; }
	}
      }
      double const top_1_acc = double(num_pos) / num_test;
      printf( "top_1_acc=%s num_pos=%s num_test=%s\n", str(top_1_acc).c_str(), str(num_pos).c_str(), str(num_test).c_str() );
    }
  };


  // FIXME: dupe'd code with display_pil_t
  struct display_lmdb_t : virtual public nesi, public lmdb_parse_datums_t // NESI(
			 // help="display caffe lmdb of images in video window",
			 // bases=["lmdb_parse_datums_t"], type_id="display_lmdb")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t rand_winds; //NESI(default=0,help="if set, display 1/2 image size random windows instead of full image")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    display_lmdb_t( void ) { }

    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_lmdb_t::on_frame, this, _1 ) ); 

      p_datum_t next_datum = read_next_datum();
      if( !next_datum ) { return; }
      p_img_t img = datum_to_img( next_datum );

      if( !rand_winds ) { disp_imgs->at(0)->fill_with_pel( grey_to_pel( 128 ) ); }
      img_copy_to_clip( img.get(), disp_imgs->at(0).get() );
      disp_win.update_disp_imgs();
    }
    void on_quit( error_code const & ec ) { get_io( &disp_win ).stop(); }

    void on_lb( error_code const & ec ) { 
      register_lb_handler( disp_win, &display_lmdb_t::on_lb, this ); // re-register handler for next event
      lb_event_t const & lbe = get_lb_event(&disp_win);
      //printf( "lbe.is_key=%s lbe.keycode=%s\n", str(lbe.is_key).c_str(), str(lbe.keycode).c_str() );
      bool unknown_command = 0;
      if( 0 ) { }
      //else if( lbe.is_key && (lbe.keycode == 'd') ) { mod_adj( cur_img_ix, img_db->img_infos.size(),  1 ); auto_adv=0; }
      //else if( lbe.is_key && (lbe.keycode == 'a') ) { mod_adj( cur_img_ix, img_db->img_infos.size(), -1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'p') ) { auto_adv ^= 1; }
      else if( lbe.is_key ) { // unknown command handlers
	unknown_command = 1; 
	printf("unknown/unhandled UI key event with keycode = %s\n", str(lbe.keycode).c_str() ); } 
      else { unknown_command = 1; printf("unknown/unhandled UI event\n"); } // unknown command
      if( !unknown_command ) { // if known command, force redisplay now
	frame_timer->cancel();
	frame_timer->expires_from_now( time_duration() );
	frame_timer->async_wait( bind( &display_lmdb_t::on_frame, this, _1 ) ); 
      }
    }

    virtual void main( nesi_init_arg_t * nia ) {
      lmdb_open_and_start_read_pass();
      disp_imgs = disp_win.disp_setup( {{300,300}} );      
      //cur_img_ix = 0;
      io_service_t & io = get_io( &disp_win );
      frame_timer.reset( new deadline_timer_t( io ) );
      frame_dur = microseconds( 1000 * 1000 / fps );
      frame_timer->expires_from_now( time_duration() );
      frame_timer->async_wait( bind( &display_lmdb_t::on_frame, this, _1 ) );
      register_quit_handler( disp_win, &display_lmdb_t::on_quit, this );
      register_lb_handler( disp_win, &display_lmdb_t::on_lb, this );
      io.run();
    }

  };

#include"gen/lmdb_caffe_io.cc.nesi_gen.cc"

}
