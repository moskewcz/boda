#include"boda_tu_base.H"
#include"img_io.H"
#include"str_util.H"
#include<turbojpeg.h>
#include<boost/iostreams/device/mapped_file.hpp>

namespace boda 
{
  using::boost::iostreams::mapped_file;
  using std::string;

  void img_t::load_fn( std::string const & fn )
  {
    ensure_is_regular_file( fn );
    if(0){}
    else if( endswith(fn,".jpg") ) { load_fn_jpeg( fn ); }
    else { rt_err( "img_t::load_fn( '%s' ): could not auto-detect file-type from extention. known extention/types are:"
		   " '.jpg':jpeg"); }
  }
  
  void check_tj_ret( int const & tj_ret, string const & err_tag ) { 
    if( tj_ret ) { rt_err( err_tag + " failed:" + string(tjGetErrorStr()) ); } }

  // note: sets row_align to a default value if it is zero
  void img_t::set_sz_and_alloc_pels( uint32_t const w_, uint32_t const h_ )
  {
    w = w_; h = h_; 
    if( (!w) || (!h) ) { 
      rt_err( strprintf("can't create zero-area image. requests WxH was %sx%s",str(w).c_str(), str(h).c_str())); }
    if( !row_align ) { row_align = sizeof( void * ); } // minimum alignment for posix_memalign
    uint32_t row_size =  depth * w; // 'unaligned' (maybe not mult of row_align) / raw / minimum
    uint32_t const ciel_row_size_over_row_align = (row_size+row_align-1)/row_align;
    row_pitch = ciel_row_size_over_row_align * row_align; // multiple of row_align / padded
    pels = ma_p_uint8_t( row_pitch*h, row_align );
  }

  void img_t::load_fn_jpeg( std::string const & fn )
  {
    mapped_file mfile( fn );
    if( !mfile.is_open() ) { rt_err( "failed to open/map file '"+fn+"' for reading" ); }
    int tj_ret = 0, jss = 0, jw = 0, jh = 0;
    tjhandle tj_dec = tjInitDecompress();
    check_tj_ret( !tj_dec, "tjInitDecompress" ); // note: !tj_dec passed as tj_ret, since 0 is the fail val for tj_dec
    uint32_t const tj_pixel_format = TJPF_RGBA;
    tj_ret = tjDecompressHeader2( tj_dec, (uint8_t *)mfile.data(), mfile.size(), &jw, &jh, &jss);
    check_tj_ret( tj_ret, "tjDecompressHeader2" );
    assert_st( (jw > 0) && ( jh > 0 ) ); // what to do with jss? seems unneeded.
    assert_st( tjPixelSize[ tj_pixel_format ] == depth );
    set_sz_and_alloc_pels( jw, jh );
    tj_ret = tjDecompress2( tj_dec, (uint8_t *)mfile.data(), mfile.size(), pels.get(), w, row_pitch, h, 
			    tj_pixel_format, 0 );
    check_tj_ret( tj_ret, "tjDecompress2" );
    tj_ret = tjDestroy( tj_dec ); 
    check_tj_ret( tj_ret, "tjDestroy" );
  }

  // for all downsample functions, scale is 0.16 fixed point, value must be in [.5,1)
  p_img_t downsample_w_transpose( img_t const * const src, uint16_t scale )
  {
    p_img_t ret( new img_t );
    ret->set_row_align( src->row_align ); // preserve alignment
    ret->set_sz_and_alloc_pels( src->h, (uint64_t( src->w ) * scale) >> 16 ); // downscale in w and transpose
    
    // FIXME: nearest sampling --> interpolate
    for( uint32_t rx = 0; rx < ret->w; ++rx ) {
      for( uint32_t ry = 0; ry < ret->h; ++ry ) {
	for( uint32_t c = 0; c < 3; ++c ) {
	  uint32_t const sx = (uint64_t( ry ) << 16) / scale;
	  ret->pels.get()[ ry*ret->row_pitch + rx*ret->depth + c ] = 
	    src->pels.get()[ rx*src->row_pitch + sx*src->depth + c ];
	}
	ret->pels.get()[ ry*ret->row_pitch + rx*ret->depth + 3 ] = uint8_t_const_max; // alpha
      }
    }
    return ret;
  }

  p_img_t img_t::downsample( uint16_t scale )
  {
    p_img_t tmp_img = downsample_w_transpose( this, scale );
    return downsample_w_transpose( tmp_img.get(), scale );
  }

};
