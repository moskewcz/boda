// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"img_io.H"
#include"timers.H"
#include<turbojpeg.h>
#include<boost/iostreams/device/mapped_file.hpp>

namespace boda 
{

  void check_tj_ret_fn( int const & tj_ret, string const & fn, string const & err_tag ) { 
    if( tj_ret ) { rt_err( "failed to load image '"+fn+"': "  + err_tag + " failed:" + string(tjGetErrorStr()) ); } }
  void check_tj_ret( int const & tj_ret, string const & err_tag ) { 
    if( tj_ret ) { rt_err( "failed to convert img_t to jpeg: "  + err_tag + " failed:" + string(tjGetErrorStr()) ); } }

  
  struct uint8_t_tj_deleter { 
    void operator()( uint8_t * const & b ) const { tjFree( b ); } // can't fail, apparently ...
  };

  void img_t::load_fn_jpeg( std::string const & fn )
  {
    p_mapped_file_source mfile = map_file_ro( fn );
    int tj_ret = 0, jss = 0, jw = 0, jh = 0;
    tjhandle tj_dec = tjInitDecompress();
    check_tj_ret_fn( !tj_dec, fn, "tjInitDecompress" ); // note: !tj_dec passed as tj_ret, since 0 is the fail val for tj_dec
    uint32_t const tj_pixel_format = TJPF_RGBA;
    tj_ret = tjDecompressHeader2( tj_dec, (uint8_t *)mfile->data(), mfile->size(), &jw, &jh, &jss);
    check_tj_ret_fn( tj_ret, fn, "tjDecompressHeader2" );
    assert_st( (jw > 0) && ( jh > 0 ) ); // what to do with jss? seems unneeded.
    assert_st( tjPixelSize[ tj_pixel_format ] == depth );
    set_sz_and_alloc_pels( i32_to_u32(i32_pt_t{jw, jh}) );
    tj_ret = tjDecompress2( tj_dec, (uint8_t *)mfile->data(), mfile->size(), pels.get(), sz.d[0], row_pitch, sz.d[1], 
			    tj_pixel_format, 0 );
    check_tj_ret_fn( tj_ret, fn, "tjDecompress2" );
    tj_ret = tjDestroy( tj_dec ); 
    check_tj_ret_fn( tj_ret, fn, "tjDestroy" );
  }

  p_uint8_with_sz_t img_t::to_jpeg( void ) {
    timer_t t("save_fn_jpeg");
    int tj_ret = -1;
    tjhandle tj_enc = tjInitCompress();
    check_tj_ret( !tj_enc, "tjInitCompress" ); // note: !tj_dec passed as tj_ret, since 0 is the fail val for tj_dec
    int const quality = 90;
    uint32_t const tj_pixel_format = TJPF_RGBA;
    ulong tj_size_out = 0;
    uint8_t * tj_buf_out = 0;
    tj_ret = tjCompress2( tj_enc, pels.get(), sz.d[0], row_pitch, sz.d[1], tj_pixel_format, &tj_buf_out, &tj_size_out, TJSAMP_444, quality, 0 );
    check_tj_ret( tj_ret, "tjCompress2" );
    assert_st( tj_size_out > 0 );
    p_uint8_with_sz_t ret( tj_buf_out, tj_size_out, uint8_t_tj_deleter() );
    tj_ret = tjDestroy( tj_enc ); 
    check_tj_ret( tj_ret, "tjDestroy" );
    return ret;
  }
  void img_t::save_fn_jpeg( std::string const & fn ) {
    // write to file
    p_uint8_with_sz_t as_jpeg = to_jpeg();
    p_ostream out = ofs_open( fn );
    bwrite_bytes( *out, (char const *)as_jpeg.get(), as_jpeg.sz );
  }

}
