// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"nesi.H"
extern "C" {
#include"libavformat/avformat.h"
#include"libavcodec/avcodec.h"
}
#include"img_io.H"

namespace boda 
{
  
  struct data_stream_ffmpeg_src_t : virtual public nesi, public data_stream_t // NESI(
                                    // help="parse file with ffmpeg (libavformat,...) output one block per raw video frame",
                                    // bases=["data_stream_t"], type_id="ffmpeg-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    
    filename_t fn; //NESI(req=1,help="input filename")
    p_dims_t out_dims; // NESI(help="set dims of output to this. size must match data size. if not set, will typically emit 'bytes' (1D array of uint8_t, with one dim named 'v'), but default may depend on specific file reader.")
    uint32_t stream_index; //NESI(default="0",help="ffmpeg stream index from which to extract frames from")

    virtual string get_pos_info_str( void ) { return string( "ffmpeg-src: pos info TODO" ); }

    virtual bool seek_to_block( uint64_t const & frame_ix ) { return false; }

    AVFormatContext *ic;
    AVCodecContext *avctx;

    data_stream_ffmpeg_src_t( void ) : ic( 0 ) { }
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      ic = avformat_alloc_context();
      if (!ic) { rt_err( "avformat_alloc_context() failed" ); }
      //ic->interrupt_callback.callback = decode_interrupt_cb;
      //ic->interrupt_callback.opaque = is;
      AVDictionary * format_opts = NULL;
      // note: by default, ffplay sets "scan_all_pmts" to 1 here. but, perhaps we can ignore that, since it's unclear if
      // it's relevant to us -- seems only to be for mpegts containers, and then only sometimes relevant?
      string const ffmpeg_url = "file:" + fn.exp;
      //AVInputFormat *iformat;
      int err;
      err = avformat_open_input(&ic, ffmpeg_url.c_str(), NULL, &format_opts);
      if( err < 0 ) { rt_err( strprintf( "avformat_open_input failed for ffmpeg_url=%s\n", str(ffmpeg_url).c_str() ) ); }
      // note: here, we could check that all options were consumed. but, we're not setting any, so why bother. see the
      // relevant check in ffplay.c
      err = avformat_find_stream_info(ic, NULL);
      if( err < 0 ) { printf( "warning: avformat_find_stream_info() failed for ffmpeg_url=%s\n", str(ffmpeg_url).c_str() ); }

      if( !( stream_index < ic->nb_streams ) ) {
        rt_err( strprintf( "user requested (zero-based) stream_index=%s, but ffmpeg says there are only ic->nb_streams=%s streams.\n",
                           str(stream_index).c_str(), str(ic->nb_streams).c_str() ) );
      }
    
      for( uint32_t i = 0; i != ic->nb_streams; ++i ) {
        // FIXME: for no obvious reason, av_dump_format() seems to print nothing -- maybe an stdio/iostreams or other
        // C++-and-ffmpeg issue?
#if 0
        printf( "av_dump_format for stream: i=%s\n", str(i).c_str() );
        av_dump_format(ic, i, ffmpeg_url.c_str(), 0);
#endif
        ic->streams[stream_index]->discard = ( i == stream_index ) ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
      }
      AVStream * const vid_st = ic->streams[stream_index];
      // FIXME/NOTE: it seems we could use either a direct check on vid_st_type or avformat_match_stream_specifier here. hmm.
      // AVMediaType vid_st_type = vid_st->codecpar->codex_type;
      int const avmss_ret = avformat_match_stream_specifier( ic, vid_st, "v" );
      assert_st( avmss_ret >= 0 );
      if( avmss_ret == 0 ) { rt_err( strprintf( "stream stream_index=%s is not a video stream", str(stream_index).c_str() ) ); }
      init_video_stream_decode( vid_st );

      // FIXME: need to close input (which calls avformat_free_context() internally)
      // avformat_close_input(&ic);

    }


    void init_video_stream_decode( AVStream * const vid_st ) {
      if( out_dims ) { return; } // for raw mode, no decoding will be done, so don't init codec (we might not be able to anyway)
      avctx = avcodec_alloc_context3(NULL);
      if (!avctx) { rt_err( "avcodec_alloc_context3() failed" ); }

      int avcodec_ret;
#if FFMPEG_31
      avcodec_ret = avcodec_parameters_to_context(avctx, vid_st->codecpar);
      if( avcodec_ret < 0 ) { rt_err( "avcodec_parameters_to_context() failed" ); }
#else
      avctx = vid_st->codec;
#endif
      av_codec_set_pkt_timebase(avctx, vid_st->time_base);

      AVCodec *codec;
      codec = avcodec_find_decoder(avctx->codec_id);

      if( !codec ) { rt_err( strprintf( "no codec could be found for id avctx->codex_id=%s\n", str(avctx->codec_id).c_str() ) ); }
      avctx->codec_id = codec->id;
      
      AVDictionary *opts = NULL;

      // this seems nice to set ... but what happens if we don't have it? for now, we die/fail.
      if(codec->capabilities & AV_CODEC_CAP_DR1) {
        avctx->flags |= CODEC_FLAG_EMU_EDGE;
      }
      else {
        rt_err( "maybe-unsupported/FIXME: codec without AV_CODEC_CAP_DR1" );
      }

      if (!av_dict_get(opts, "threads", NULL, 0)) {
        av_dict_set(&opts, "threads", "auto", 0);
      }
      
      av_dict_set(&opts, "refcounted_frames", "1", 0);
      avcodec_ret = avcodec_open2(avctx, codec, &opts);
      if( avcodec_ret < 0 ) { rt_err( "avcodec_open2() failed" ); }

      // check for any unconsume (unrecognized) options
      AVDictionaryEntry *t = NULL;
      if ((t = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX))) {
        rt_err( strprintf( "unknown code option '%s'\n", t->key ) );
      }

      // FIXME: use shared_ptr deleter (or whatever) to dealloc these
      // avcodec_free_context(&avctx); // only if FFMPEG_31
      // av_dict_free(&opts);
    }

    zi_uint64_t frame_ix;
    virtual data_block_t proc_block( data_block_t const & db ) {
      assert_st( ic );
      data_block_t ret = db;
      AVPacket pkt;
      int const err = av_read_frame(ic, &pkt);
      if( err < 0 ) { return ret; }
      assert_st( (uint32_t)pkt.stream_index == stream_index ); // AVDISCARD_ALL setting for other streams in init() should guarentee this
      ret.nda = make_shared<nda_t>( dims_t{ vect_uint32_t{uint32_t(pkt.size)}, vect_string{ "v" }, "uint8_t" } );
      std::copy( pkt.data, pkt.data + pkt.size, (uint8_t *)ret.d() );
      if( out_dims ) { // raw mode
        assert_st( ret.nda );
        ret.nda->reshape( *out_dims );
        ret.frame_ix = frame_ix.v;
        ++frame_ix.v;
        return ret;
      }

      AVFrame *frame = av_frame_alloc();
      int got_frame = 0;
      int const decode_ret = avcodec_decode_video2(avctx, frame, &got_frame, &pkt);
      if( decode_ret < 0 ) { rt_err("avcodec_decode_video2() failed"); }
      if( decode_ret != pkt.size ) { rt_err("decode didn't consume entire packet?"); }

      if( got_frame ) {
        if( frame->format != AV_PIX_FMT_YUV420P ) {
          rt_err( "only the AV_PIX_FMT_YUV420P pixel format is currently supported for decoded output (adding conversions is TODO)" );
        }
        if( (frame->width&1) || (frame->height&1) ) {
          rt_err( strprintf( "only even frame sizes are supported, but frame->width=%s frame->height=%s\n",
                             str(frame->width).c_str(), str(frame->height).c_str() ) );
        }
        // convert YUV planes to data block
        vect_p_nda_uint8_t yuv_ndas;
        ret.subblocks = make_shared<vect_data_block_t>();      
        for( uint32_t pix = 0; pix != 3; ++pix ) {
          uint32_t const subsample = pix ? 2 : 1;
          string const meta = string("YUV_") + string("YUV")[pix];
          uint32_t const ph = frame->height/subsample;
          uint32_t const pw = frame->width/subsample;
          p_nda_uint8_t yuv_nda = make_shared<nda_uint8_t>( dims_t{ vect_uint32_t{uint32_t(ph), uint32_t(pw)}, vect_string{ "y","x" },"uint8_t" });
          // fill in y,u, or v data
          for( uint32_t y = 0; y != ph; ++y ) {
            uint8_t * rb = frame->data[pix] + frame->linesize[pix]*y;
            std::copy( rb, rb + pw, &yuv_nda->at1(y) );
          }
          yuv_ndas.push_back( yuv_nda );
          if( pix == 0 ) {
            ret.meta = meta;
            ret.nda = yuv_nda;
          } else {
            data_block_t uv_db;
            uv_db.meta = meta;
            uv_db.nda = yuv_nda;
            ret.subblocks->push_back( uv_db );
          }
        }
        ret.as_img = make_shared< img_t >();
        ret.as_img->set_sz_and_pels_from_yuv_420_planes( yuv_ndas );
        ret.frame_ix = frame_ix.v;
        ++frame_ix.v;
      } else {
        ret.need_more_in = 1;
      }      
      av_frame_free(&frame);
      return ret;
    }

  };
  
#include"gen/data-stream-ffmpeg.cc.nesi_gen.cc"

}
