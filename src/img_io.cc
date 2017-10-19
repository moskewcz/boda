// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"img_io.H"
#include"pyif.H"
#include"str_util.H"
#include"ext/lodepng.h"
#include<boost/iostreams/device/mapped_file.hpp>
#include"has_main.H"
#include"timers.H"
// #include<omp.h> // not currently needed (even when OpenMP is used)

namespace boda 
{
  void img_t::load_fn( std::string const & fn )
  {
    ensure_is_regular_file( fn );
    if(0){}
    else if( endswith(fn,".jpg") ) { load_fn_jpeg( fn ); }
    else if( endswith(fn,".png") ) { load_fn_png( fn ); }
    else { rt_err( "failed to load image '"+fn+"': could not auto-detect file-type from extention."
		   " known extention/types are:"
		   " '.jpg':jpeg '.png':png"); }
  }
  
  // note: sets row_align to a default value if it is zero
  void img_t::set_sz( u32_pt_t const & sz_ ) {
    sz = sz_;
    if( !sz.both_dims_non_zero() ) {
      rt_err( strprintf("can't create zero-area image. requests WxH was %sx%s",str(sz.d[0]).c_str(), str(sz.d[1]).c_str())); }
    if( !row_align ) { row_align = sizeof( void * ); } // minimum alignment for posix_memalign
    uint32_t row_size =  depth * sz.d[0]; // 'unaligned' (maybe not mult of row_align) / raw / minimum
    uint32_t const ceil_row_size_over_row_align = (row_size+row_align-1)/row_align;
    row_pitch = ceil_row_size_over_row_align * row_align; // multiple of row_align / padded
    row_pitch_pels = row_pitch / depth;
    assert_st( row_pitch_pels * depth == row_pitch ); // could relax?
  }
  void img_t::alloc_pels( void ) { pels = ma_p_uint8_t( sz_raw_bytes(), row_align ); }
  void img_t::set_sz_and_alloc_pels( u32_pt_t const & sz_ ) { set_sz( sz_ ); alloc_pels(); }

  
  // note: this is a bit hacky/limited: only the sz and yuv_pels fields of the img will be set. also, no padding/stride
  // is supported for the planes (but maybe it could/should be).
  void img_t::set_sz_and_pels_from_yuv_420_planes( vect_p_nda_uint8_t const & yuv_ndas ) {
    assert_st( yuv_ndas.size() == 3 );
    for( uint32_t pix = 0; pix != 3; ++pix ) {
      yuv_pels.push_back( yuv_ndas[pix]->get_internal_data() );
      u32_pt_t yuv_sz = get_xy_dims_strict( yuv_ndas[pix]->dims );
      if( !pix ) {
        sz = yuv_sz;
        // note: we don't set row_align, row_pitch, row_pitch_pels (leaving them at zero), since we don't set pels and
        // should not use them.
      } else {
        // FIXME/NOTE: could relax. per dim, if size of Y plane is odd, sizes of UV places should be ceil_div-by-2 of it.
        if( yuv_sz.scale(2) != sz ) { rt_err( strprintf( "yuv plane size mismatch. Y sz=%s, but for pix=%s, yuv_sz=%s\n",
                                                         str(sz).c_str(), str(pix).c_str(), str(yuv_sz).c_str() ) ); }
      }
    }
    
  }
  
  void img_t::fill_with_pel( uint32_t const & v ) {
    uint64_t const vv = (uint64_t(v) << 32) + v;
    uint64_t * const dest_data = (uint64_t *) pels.get(); 
    assert_st( !(row_pitch_pels&1) );
#pragma omp parallel for
    for( uint32_t i = 0; i < (row_pitch_pels>>1)*sz.d[1]; ++i ) { dest_data[i] = vv; }
  }
  
  void img_t::load_fn_jpeg( std::string const & fn ) {
    p_mapped_file_source mfile = map_file_ro( fn );
    p_uint8_with_sz_t mfile_data( mfile, (uint8_t *)mfile->data(), mfile->size() ); // alias ctor to bind lifetime to mapped file
    from_jpeg( mfile_data, fn );
  }

  void img_t::load_fn_png( std::string const & fn )
  {
    p_mapped_file_source mfile = map_file_ro( fn );
    uint32_t const lp_depth = 4;
    assert( depth == lp_depth );
    vect_uint8_t lp_pels; // will contain packed RGBA (no padding)
    unsigned lp_w = 0, lp_h = 0;
    unsigned ret = lodepng::decode( lp_pels, lp_w, lp_h, (uint8_t *)mfile->data(), mfile->size() );
    if( ret ) { rt_err( strprintf( "failed to load image '%s': lodepng decoder error %s: %s", 
				   fn.c_str(), str(ret).c_str(), lodepng_error_text(ret) ) ); }
    assert_st( (lp_w > 0) && ( lp_h > 0 ) );
    set_sz_and_alloc_pels( {lp_w, lp_h} );
    // copy packed data into our (maybe padded) rows
    uint32_t const pack_rl = sz.d[0]*lp_depth;
    for( uint32_t i = 0; i < sz.d[1]; ++i ) { memcpy( pels.get() + i*row_pitch, (&lp_pels[0]) + i*pack_rl, pack_rl ); }
  }

  void img_t::save_fn_png( std::string const & fn, bool const disable_compression )
  {
    timer_t t("save_fn_png");

    uint32_t const lp_depth = 4;
    assert( depth == lp_depth );
    vect_uint8_t lp_pels; // will contain packed RGBA (no padding)
    lp_pels.resize( sz.dims_prod()*4 );
    // copy our (maybe padded) rows into packed data
    uint32_t const pack_rl = sz.d[0]*lp_depth;
    for( uint32_t i = 0; i < sz.d[1]; ++i ) { memcpy( (&lp_pels[0]) + i*pack_rl, pels.get() + i*row_pitch, pack_rl ); }
    vect_uint8_t png_file_data;
    
    lodepng::State state;
    if( disable_compression ) { state.encoder.zlibsettings.btype=0; }
    unsigned ret = lodepng::encode( png_file_data, lp_pels, sz.d[0], sz.d[1], state );
    if( ret ) { rt_err( strprintf( "failed to encode image '%s': lodepng decoder error %s: %s", 
				   fn.c_str(), str(ret).c_str(), lodepng_error_text(ret) ) ); }
    lodepng::save_file(png_file_data, fn);
  }

  // like a swap: no data copy *and* preserves img_t object identity. but, changes internal pixel data pointer
  void img_t::share_pels_from( p_img_t o ) {
    assert_st( o->depth == depth );
    assert_st( o->sz == sz );
    assert_st( o->row_pitch == row_pitch );
    pels = o->pels;
  }

  p_img_t transpose( img_t const * const src )
  {
    p_img_t ret( new img_t );
    ret->set_row_align( src->row_align ); // preserve alignment
    ret->set_sz_and_alloc_pels( {src->sz.d[1], src->sz.d[0]} );
    for( uint32_t rx = 0; rx < ret->sz.d[0]; ++rx ) {
      for( uint32_t ry = 0; ry < ret->sz.d[1]; ++ry ) {
	for( uint32_t c = 0; c < 4; ++c ) {
	  uint32_t const sx = ry;
	  uint32_t const sy = rx;
	  ret->pels.get()[ ry*ret->row_pitch + rx*ret->depth + c ] =
	    src->pels.get()[ sy*src->row_pitch + sx*src->depth + c ];
	}
      }
    }
    return ret;
  }

  p_img_t downsample_w_transpose_2x( img_t const * const src )
  {
    p_img_t ret( new img_t );
    ret->set_row_align( src->row_align ); // preserve alignment
    ret->set_sz_and_alloc_pels( {src->sz.d[1], (src->sz.d[0]+1) >> 1} ); // downscale in w and transpose. 
    // note: if src->sz.d[0] is odd, we will just copy the last input column
    bool const src_w_odd = (src->sz.d[0]&1);
    uint64_t const * src_data = (uint64_t const *) src->pels.get(); 
    uint32_t const max_alpha = 0xffu << (3*8);
#pragma omp parallel for 
    for( uint32_t ry = 0; ry < ret->sz.d[1]-src_w_odd; ++ry ) {
      uint32_t const sx = ry << 1;
      for( uint32_t rx = 0; rx < ret->sz.d[0]; ++rx ) {
	uint32_t const sy = rx;
	uint64_t const src_data_1 = src_data[ (sy*src->row_pitch_pels + sx)>>1 ];
	uint32_t const src_data_2 = src_data_1 >> 32;
	uint32_t dest_val = max_alpha;
	for( uint32_t c = 0; c < 3; ++c ) {
	  dest_val += // note: we round .5 up
	    uint32_t( uint8_t( (
				 uint16_t( get_chan(c,src_data_1) ) +
				 uint16_t( get_chan(c,src_data_2) ) + 1 ) >> 1 ) ) << (c*8);
	}
	((uint32_t *)ret->pels.get())[ ry*ret->row_pitch_pels + rx ] = dest_val;
      }
    }
    if( src_w_odd ) {
      for( uint32_t rx = 0; rx < ret->sz.d[0]; ++rx ) {
	uint32_t const sy = rx;
	((uint32_t *)ret->pels.get())[ (ret->sz.d[1]-1)*ret->row_pitch_pels + rx ] =
	  ((uint32_t *)src->pels.get())[ sy*src->row_pitch_pels + (src->sz.d[0]-1) ] | max_alpha;
      }
    }
    return ret;
  }

  p_img_t downsample_2x( p_img_t img ) {
    timer_t ds_timer("downsample_2x");
    p_img_t tmp_img = downsample_w_transpose_2x( img.get() );
    return downsample_w_transpose_2x( tmp_img.get() );
  }

  // downsample by 2x in both dims as many times as possible without
  // yielding an image smaller than is desired.
  p_img_t downsample_2x_to_size( p_img_t img, u32_pt_t const & ds ) {
    timer_t ds_timer("downsample_2x_to_size");
    assert_st( ds.both_dims_non_zero() );
    while( 1 ) {
      u32_pt_t const ds_sz = (img->sz + u32_pt_t{1,1}) >> 1; 
      bool const do_x_ds = ds_sz.d[0] >= ds.d[0];
      bool const do_y_ds = ds_sz.d[1] >= ds.d[1];
      if( !(do_x_ds || do_y_ds) ) { return img; }
      if( do_x_ds ) { img = downsample_w_transpose_2x( img.get() ); } else { img = transpose( img.get() ); }
      if( do_y_ds ) { img = downsample_w_transpose_2x( img.get() ); } else { img = transpose( img.get() ); }
    }
  }

//#define RESIZE_DEBUG
  // for all downsample functions, scale is 0.16 fixed point, value must be in [.5,1)
  p_img_t downsample_w_transpose( img_t const * const src, uint32_t ds_w )
  {
    assert( ds_w );
    assert( ds_w <= src->sz.d[0] ); // scaling must be <= 1
    if( ds_w == src->sz.d[0] ) { return transpose( src ); } // special case for no scaling (i.e. 1x downsample)
    assert( (ds_w<<1) >= src->sz.d[0] ); // scaling must be >= .5
    if( (ds_w<<1) == src->sz.d[0] ) { return downsample_w_transpose_2x( src ); } // special case for 2x downsample
    p_img_t ret( new img_t );
    ret->set_row_align( src->row_align ); // preserve alignment
    ret->set_sz_and_alloc_pels( {src->sz.d[1], ds_w} ); // downscale in w and transpose
    uint16_t scale = (uint64_t(ds_w)<<16)/src->sz.d[0]; // 0.16 fixed point
    uint32_t inv_scale = (uint64_t(1)<<46)/scale; // 2.30 fixed point, value (1,2]
    // for clamping sx2_fp, which (due to limitied precision of scale)
    // may exceed this value (which would cause sampling off the edge
    // of the src image). the dropped (impossible) sample should have
    // near-zero weight (again bounded by the precision of scale and
    // the input and output image sizes).
    uint64_t const max_src_x = uint64_t(src->sz.d[0])<<30; 
    //printf( "scale=%s inv_scale=%s ((double)scale/double(1<<16))=%s ((double)inv_scale/double(1<<30))=%s\n", str(scale).c_str(), str(inv_scale).c_str(), str(((double)scale/double(1<<16))).c_str(), str(((double)inv_scale/double(1<<30))).c_str() );
    uint32_t const * src_data = (uint32_t const *) src->pels.get(); 
    assert( src->depth == 4 );
#pragma omp parallel for
    for( uint32_t ry = 0; ry < ret->sz.d[1]; ++ry ) {
      uint64_t const sx1_fp = (uint64_t( ry ) * inv_scale); // img_sz_bits.30 fp
      uint32_t const sx1 = sx1_fp >> 30;
      uint64_t const sx2_fp = std::min( max_src_x, sx1_fp + inv_scale );
      uint32_t const sx2 = sx2_fp >> 30;
      uint64_t const sx1_w = ( 1U << 30 ) - (sx1_fp - (uint64_t(sx1)<<30)); // 1.30 fp, value (0,1]
      uint64_t const sx2_w = sx2_fp - (uint64_t(sx2)<<30); // 1.30 fp, value (0,1]
      uint32_t const span = sx2 - sx1;
      assert( (span == 1) || (span == 2) );
#ifdef RESIZE_DEBUG
      printf("------\n");
      printf( "ry=%s sx1=%s sx2=%s\n", str(ry).c_str(), str(sx1).c_str(), str(sx2).c_str() );
      printf( "sx1=%s sx2=%s sx1_w=%s sx2_w=%s\n", str(sx1).c_str(), str(sx2).c_str(), str(sx1_w).c_str(), str(sx2_w).c_str() );
      printf( "(double(sx1_fp)/double(1<<30))=%s\n", str((double(sx1_fp)/double(1U<<30))).c_str() );
      printf( "(double(sx2_fp)/double(1<<30))=%s\n", str((double(sx2_fp)/double(1U<<30))).c_str() );
      
      printf( "(double(sx1_w)/double(1<<30))=%s\n", str((double(sx1_w)/double(1U<<30))).c_str() );
      printf( "(double(sx2_w)/double(1<<30))=%s\n", str((double(sx2_w)/double(1U<<30))).c_str() );
#endif	
      for( uint32_t rx = 0; rx < ret->sz.d[0]; ++rx ) {
	uint32_t const src_data_1 = src_data[ rx*src->row_pitch_pels + sx1 ];
	uint32_t const src_data_m = (span==1)?0:src_data[ rx*src->row_pitch_pels + (sx1+1) ];
	uint32_t const src_data_2 = sx2_w ? src_data[ rx*src->row_pitch_pels + sx2 ] : 0;

	uint32_t dest_val = 0xffu << (3*8); // max alpha
	for( uint32_t c = 0; c < 3; ++c ) {
	  dest_val += 
	    uint32_t( uint8_t( ( (
			 ( get_chan_64(c,src_data_1) * sx1_w ) +
			 ( (get_chan_64(c,src_data_m) << 30) ) + // note: src_data_m may be zero
			 ( get_chan_64(c,src_data_2) * sx2_w )
				  ) * scale + (1lu << 45) ) >> 46 ) ) << (c*8);
#ifdef RESIZE_DEBUG
	  uint32_t p1 = src->pels.get()[ rx*src->row_pitch + sx1*src->depth + c ];
	  uint32_t p2 = src->pels.get()[ rx*src->row_pitch + (sx1+1)*src->depth + c ];
	  uint32_t p3 = src->pels.get()[ rx*src->row_pitch + sx2*src->depth + c ];
	  uint32_t r = ret->pels.get()[ ry*ret->row_pitch + rx*ret->depth + c ];
	  printf( "p1=%s p2=%s p3=%s r=%s\n", str(p1).c_str(), str(p2).c_str(), str(p3).c_str(), str(r).c_str() );
#endif
	  assert( (!sx2_w) || (sx2 < src->sz.d[0]) );
	}
	((uint32_t *)ret->pels.get())[ ry*ret->row_pitch_pels + rx ] = dest_val;
      }
    }
    return ret;
  }

  p_img_t downsample_up_to_2x_to_size( p_img_t img, u32_pt_t const & ds ) { // ds_w must be in [ceil(w/2),w]
    timer_t ds_timer("downsample_up_to_2x_to_size");
    assert( ds.d[0] <= img->sz.d[0] );
    assert( ds.d[0] >= ((img->sz.d[0]+1)>>1) );
    assert( ds.d[1] <= img->sz.d[1] );
    assert( ds.d[1] >= ((img->sz.d[1]+1)>>1) );
    p_img_t tmp_img = downsample_w_transpose( img.get(), ds.d[0] );
    return downsample_w_transpose( tmp_img.get(), ds.d[1] );    
  }

  p_img_t downsample_up_to_2x( p_img_t img, double const scale ) { 
    timer_t ds_timer("downsample");
    assert( scale <= 1 );
    assert( scale >= .5 );
    if( scale == 1 ) { return img; }
    return downsample_up_to_2x_to_size( img, img->sz.scale_and_round( scale ) );
  }

  p_img_t downsample_to_size( p_img_t img, u32_pt_t const & ds ) { 
    return downsample_up_to_2x_to_size( downsample_2x_to_size( img, ds ), ds ); }

  string img_t::WxH_str( void ) { return strprintf( "%sx%s", str(sz.d[0]).c_str(), str(sz.d[1]).c_str()); }

  void downsample_test( string const & fn )
  {
    timer_t ds_timer("ds_test");
    p_img_t img( new img_t );
    img->load_fn( fn.c_str() );
    p_img_t cur = img;
    for( uint32_t s = 0; s < 16; ++s )
    {
      if( 0 ) {
	timer_t t("py_img_show");
	py_img_show( cur, "out/scale_" + str(s) + ".png" );
      }
      if( (cur->sz.d[0] < 2) || (cur->sz.d[1] < 2) ) { break; }
      cur = downsample_2x( cur );
    }
  }

  struct ds_test_t : virtual public nesi, public has_main_t // NESI(help="run image downsampling test on a single image file",bases=["has_main_t"], type_id="ds_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string image_fn; //NESI(help="input: image filename",req=1)
    virtual void main( nesi_init_arg_t * nia ) { downsample_test( image_fn ); }
  };

  p_img_t upsample_w_transpose_2x( img_t const * const src )
  {
    p_img_t ret( new img_t );
    ret->set_row_align( src->row_align ); // preserve alignment
    ret->set_sz_and_alloc_pels( {src->sz.d[1], src->sz.d[0] << 1} ); // upscale in w and transpose. 
    uint32_t const * src_data = (uint32_t const *)src->pels.get(); 
#pragma omp parallel for
    for( uint32_t sx = 0; sx < src->sz.d[0]; ++sx ) {
      uint32_t const ry = sx << 1;
      for( uint32_t rx = 0; rx < ret->sz.d[0]; ++rx ) {
	uint32_t const sy = rx;
	uint32_t const pel = src_data[ sy*src->row_pitch_pels + sx ];
	((uint32_t *)ret->pels.get())[ (ry+0)*ret->row_pitch_pels + rx ] = pel;
	((uint32_t *)ret->pels.get())[ (ry+1)*ret->row_pitch_pels + rx ] = pel;
      }
    }
    return ret;
  }

  // upsample by 2x in both dims as many times as needed until the
  // image is at least the requested size in each dim.
  p_img_t upsample_2x_to_size( p_img_t img, u32_pt_t const & ds ) {
    timer_t ds_timer("upsample_2x_to_size");
    assert_st( ds.both_dims_non_zero() );
    while( 1 ) {
      bool const do_x_us = img->sz.d[0] < ds.d[0];
      bool const do_y_us = img->sz.d[1] < ds.d[1];
      if( !(do_x_us || do_y_us) ) { return img; }
      if( do_x_us ) { img = upsample_w_transpose_2x( img.get() ); } else { img = transpose( img.get() ); }
      if( do_y_us ) { img = upsample_w_transpose_2x( img.get() ); } else { img = transpose( img.get() ); }
    }
  }

  p_img_t resample_to_size( p_img_t img, u32_pt_t const & ds ) {
    return downsample_to_size( upsample_2x_to_size( img, ds ), ds ); 
  }    

  // copy rectangular region from src to dest, starting from src_pt in src and being placed at
  // dest_pt in dest. src_pt and dest_pt must be strictly less than thier respective image sizes
  // (i.e. must be valid). the copied size will be as large as possible such that the src/dest
  // rectangles do not go outside src/dest, but no larger than max_copy_sz.

  void img_copy_to_clip( img_t const * const src, img_t * const dest, u32_pt_t const & dest_pt,
			 u32_pt_t const & src_pt, u32_pt_t const & max_copy_sz ) {
    timer_t t("img_copy_to");
    uint32_t * const dest_data = dest->get_pel_addr( dest_pt ); 
    uint32_t const * const src_data = src->get_pel_addr( src_pt );
    u32_pt_t const l_xy = min( max_copy_sz, min( dest->sz - dest_pt, src->sz - src_pt ) );
    for( uint32_t sy = 0; sy < l_xy.d[1]; ++sy ) {
      for( uint32_t sx = 0; sx < l_xy.d[0]; ++sx ) {
	uint32_t const pel = src_data[ sy*src->row_pitch_pels + sx ];
	dest_data[ sy*dest->row_pitch_pels + sx ] = pel;
      }
    }
  }

  p_img_t upsample_2x( p_img_t img ) {
    timer_t ds_timer("upsample_2x");
    p_img_t tmp_img = upsample_w_transpose_2x( img.get() );
    return upsample_w_transpose_2x( tmp_img.get() );
  }

  // note: i==0 -> returns ic, i==num --> returns ec, otherwise returns mix: (ic*(num-i) + ec*i)/num
  uint32_t interp_pel( uint32_t const & ic, uint32_t const & ec, uint32_t const & num, uint32_t const & i ) {
    assert_st( i <= num );
    uint32_t ret = 0;
    for( uint32_t c = 0; c < 4; ++c ) {
      uint8_t const cs = c * 8; // shift amt for this channel
      uint32_t const icc = (ic >> cs) & 0xff;
      uint32_t const ecc = (ec >> cs) & 0xff;
      uint8_t const rc = uint8_t( (icc*(num-i) + ecc*(i)) / num );
      ret += rc << cs;
    }
    return ret;
  }
  // note: draws num pels. first pel = ic. last pel = (ic + ec*(num-1))/num (i.e. almost ec, next pel would be ec)
  void img_draw_pels( img_t * const dest, u32_pt_t const & d, uint32_t const & num, 
		      i32_pt_t const & stride, uint32_t const & ic, uint32_t const & ec )
  {
    assert_st( num ); // allow num == 0? if so, return here i guess ... or fix assumptions below
    u32_pt_t const fin_pel = i32_to_u32( u32_to_i32(d) + stride.scale(num - 1) );
    assert_st( fin_pel.both_dims_lt( dest->sz ) );
    uint32_t * dest_data = dest->get_pel_addr( d );
    int32_t const stride_pels = stride.d[1]*dest->row_pitch_pels + stride.d[0];
    for( uint32_t i = 0; i < num; ++i ) { *dest_data = interp_pel(ic,ec,num,i); dest_data += stride_pels; }
  }
  
#include"gen/img_io.cc.nesi_gen.cc"

};
