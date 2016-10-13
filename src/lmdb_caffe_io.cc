// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"lmdb_caffe_io.H"

#include"caffeif.H" // only test_lmdb_t

namespace boda 
{

  uint64_t score_batch( p_nda_float_t const & out_batch, vect_uint32_t const & batch_labels_gt ) {
    uint64_t num_pos = 0;
    assert( out_batch->dims.sz() == 4 );
    assert( out_batch->dims.dims(0) >= batch_labels_gt.size() );
    uint32_t const num_out = out_batch->dims.dims(1);
    //assert( num_out == 1000 ); // for imagenet
    assert( out_batch->dims.dims(2) == 1 );
    assert( out_batch->dims.dims(3) == 1 );
    for( uint32_t i = 0; i != batch_labels_gt.size(); ++i ) {
      uint32_t max_chan_ix = uint32_t_const_max;
      float max_chan_val = 0;
      for( uint32_t j = 0; j != out_batch->dims.dims(1); ++j ) {
	float const & v = out_batch->at4(i,j,0,0);
	if( (max_chan_ix == uint32_t_const_max) || (v > max_chan_val) ) { max_chan_ix = j; max_chan_val = v; }
      }
      if( !( batch_labels_gt[i] < num_out ) ) {
	rt_err( strprintf( "gt output index too large for number of network outputs: "
			   "i=%s batch_labels_gt[i]=%s num_out=%s -- gt data / network mismatch?\n", 
			   str(i).c_str(), str(batch_labels_gt[i]).c_str(), str(num_out).c_str() ) );
      }
      if( batch_labels_gt[i] == max_chan_ix ) { ++num_pos; }
    }
    return num_pos;
  }

  struct test_lmdb_t : virtual public nesi, public lmdb_parse_datums_t // NESI(
			 // help="test lmdb with run_cnet_t",
			 // bases=["lmdb_parse_datums_t"], type_id="test_lmdb")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_run_cnet_t run_cnet; //NESI(default="()",help="cnet running options")

    void main( nesi_init_arg_t * nia ) { 
      run_cnet->setup_cnet( nia ); 
      lmdb_open_and_start_read_pass();
      vect_uint32_t batch_labels_gt;
      uint64_t num_test = 0;
      uint64_t num_pos = 0;
      while( 1 ) {
	batch_labels_gt.clear();
	read_batch_of_datums( run_cnet->in_batch, batch_labels_gt );
	if( batch_labels_gt.empty() ) { break; } // quit if we run out of data early
	p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
	num_test += batch_labels_gt.size();
	num_pos += score_batch( out_batch, batch_labels_gt );
      }
      double const top_1_acc = double(num_pos) / num_test;
      printf( "top_1_acc=%s num_pos=%s num_test=%s\n", str(top_1_acc).c_str(), str(num_pos).c_str(), str(num_test).c_str() );
    }
  };

#include"gen/lmdb_caffe_io.H.nesi_gen.cc"
#include"gen/lmdb_caffe_io.cc.nesi_gen.cc"

}
