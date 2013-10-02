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
  void img_t::load_fn_jpeg( std::string const & fn )
  {
    if( !row_align ) { row_align = sizeof( void * ); } // minimum alignment for posix_memalign
    mapped_file mfile( fn );
    if( !mfile.is_open() ) { rt_err( "failed to open/map file '"+fn+"' for reading" ); }

    int tj_ret = 0, jss = 0, jw = 0, jh = 0;
    tjhandle tj_dec = tjInitDecompress();
    if( !tj_dec ) { rt_err( "tjInitDecompress failed" + string(tjGetErrorStr()) ); }

    uint32_t const tj_pixel_format = TJPF_RGBA;
    tj_ret = tjDecompressHeader2( tj_dec, (uint8_t *)mfile.data(), mfile.size(), &jw, &jh, &jss);
    if( tj_ret ) { rt_err( "tjDecompressHeader2 failed:" + string(tjGetErrorStr()) ); }
    assert_st( (jw > 0) && ( jh > 0 ) );
    w = jw; h = jh; // what to do with jss? seems unneeded.
    uint32_t row_size = tjPixelSize[ tj_pixel_format ] * w; // 'unaligned' (maybe not mult of row_align) / raw / minimum
    uint32_t const ciel_row_size_over_row_align = (row_size+row_align-1)/row_align;
    row_pitch = ciel_row_size_over_row_align * row_align; // multiple of row_align / padded
    pels = ma_p_uint8_t( row_pitch*h, row_align );

    tj_ret = tjDecompress2( tj_dec, (uint8_t *)mfile.data(), mfile.size(), pels.get(), w, row_pitch, h, 
			    tj_pixel_format, 0 );
    if( tj_ret ) { rt_err( "tjDecompress2 failed:" + string(tjGetErrorStr()) ); }

    tj_ret = tjDestroy( tj_dec );
    if( tj_ret ) { rt_err( "tjDestroy failed:" + string(tjGetErrorStr()) ); }
  }
};
