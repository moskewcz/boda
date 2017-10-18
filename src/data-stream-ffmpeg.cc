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
#include<turbojpeg.h>

namespace boda 
{
  void check_tj_ret( int const & tj_ret, string const & err_tag );

  struct uint8_t_tj_deleter { 
    void operator()( uint8_t * const & b ) const { tjFree( b ); } // can't fail, apparently ...
  };

  p_uint8_with_sz_t avframe_to_jpeg( AVFrame * const frame ) {
    int tj_ret = -1;
    tjhandle tj_enc = tjInitCompress();
    check_tj_ret( !tj_enc, "tjInitCompress" ); // note: !tj_dec passed as tj_ret, since 0 is the fail val for tj_dec
    int const quality = 90;
    ulong tj_size_out = 0;
    uint8_t * tj_buf_out = 0;
    tj_ret = tjCompressFromYUVPlanes( tj_enc, frame->data, frame->width, frame->linesize, frame->height, TJSAMP_420,
                                      &tj_buf_out, &tj_size_out, quality, 0 );
    check_tj_ret( tj_ret, "tjCompressFromYUVPlanes" );
    assert_st( tj_size_out > 0 );
    p_uint8_with_sz_t ret( tj_buf_out, tj_size_out, uint8_t_tj_deleter() );
    tj_ret = tjDestroy( tj_enc ); 
    check_tj_ret( tj_ret, "tjDestroy" );
    return ret;
  }

  
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
      if( out_dims ) { assert_st( ret.nda ); ret.nda->reshape( *out_dims ); }

      AVFrame *frame = av_frame_alloc();
      int got_frame = 0;
      int const decode_ret = avcodec_decode_video2(avctx, frame, &got_frame, &pkt);
      if( decode_ret < 0 ) { rt_err("avcodec_decode_video2() failed"); }
      if( decode_ret != pkt.size ) { rt_err("decode didn't consume entire packet?"); }

      if( got_frame ) {
        // FIXME: actually output frame
        printf( "got frame: format=%s width=%s height=%s\n", str(frame->format).c_str(), str(frame->width).c_str(), str(frame->height).c_str() );

        if( frame_ix.v < 10 ) {
          string const fn = "out-"+str(frame_ix.v)+".jpg";
          p_uint8_with_sz_t as_jpeg = avframe_to_jpeg( frame );
          p_ostream out = ofs_open( fn );
          bwrite_bytes( *out, (char const *)as_jpeg.get(), as_jpeg.sz );
        }
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
