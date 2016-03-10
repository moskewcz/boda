// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"img_io.H"
#include<turbojpeg.h>
#include<boost/iostreams/device/mapped_file.hpp>

namespace boda 
{

  void check_tj_ret( int const & tj_ret, string const & fn, string const & err_tag ) { 
    if( tj_ret ) { rt_err( "failed to load image '"+fn+"': "  + err_tag + " failed:" + string(tjGetErrorStr()) ); } }

  void img_t::load_fn_jpeg( std::string const & fn )
  {
    p_mapped_file_source mfile = map_file_ro( fn );
    int tj_ret = 0, jss = 0, jw = 0, jh = 0;
    tjhandle tj_dec = tjInitDecompress();
    check_tj_ret( !tj_dec, fn, "tjInitDecompress" ); // note: !tj_dec passed as tj_ret, since 0 is the fail val for tj_dec
    uint32_t const tj_pixel_format = TJPF_RGBA;
    tj_ret = tjDecompressHeader2( tj_dec, (uint8_t *)mfile->data(), mfile->size(), &jw, &jh, &jss);
    check_tj_ret( tj_ret, fn, "tjDecompressHeader2" );
    assert_st( (jw > 0) && ( jh > 0 ) ); // what to do with jss? seems unneeded.
    assert_st( tjPixelSize[ tj_pixel_format ] == depth );
    set_sz_and_alloc_pels( i32_to_u32(i32_pt_t{jw, jh}) );
    tj_ret = tjDecompress2( tj_dec, (uint8_t *)mfile->data(), mfile->size(), pels.get(), sz.d[0], row_pitch, sz.d[1], 
			    tj_pixel_format, 0 );
    check_tj_ret( tj_ret, fn, "tjDecompress2" );
    tj_ret = tjDestroy( tj_dec ); 
    check_tj_ret( tj_ret, fn, "tjDestroy" );
  }

}
