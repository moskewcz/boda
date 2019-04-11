// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"data-stream-file.H"
#include"nesi.H"
#include<netinet/ip.h>
#include<netinet/udp.h>

namespace boda 
{

  unsigned short in_cksum(unsigned short *addr, int len)
  {
    int				nleft = len;
    int				sum = 0;
    unsigned short	*w = addr;
    unsigned short	answer = 0;

    /*
     * Our algorithm is simple, using a 32 bit accumulator (sum), we add
     * sequential 16 bit words to it, and at the end, fold back all the
     * carry bits from the top 16 bits into the lower 16 bits.
     */
    while (nleft > 1)  {
      sum += *w++;
      nleft -= 2;
    }

    /* 4mop up an odd byte, if necessary */
    if (nleft == 1) {
      *(unsigned char *)(&answer) = *(unsigned char *)w ;
      sum += answer;
    }

    /* 4add back carry outs from top 16 bits to low 16 bits */
    sum = (sum >> 16) + (sum & 0xffff);	/* add hi 16 to low 16 */
    sum += (sum >> 16);			/* add carry */
    answer = ~sum;				/* truncate to 16 bits */
    return(answer);
  }


  // from libpcap documentation, we learned the pcap file format global and packet header format: https://wiki.wireshark.org/Development/LibpcapFileFormat
  uint32_t const pcap_file_magic = 0xa1b2c3d4;
  uint16_t const ethertype_ipv4 = 0x0800;
  
  struct ethernet_header {
    uint8_t src_mac[6];
    uint8_t dest_mac[6];
    uint16_t ethertype;
  } __attribute__(( packed ));
  
  struct pcap_hdr_t {
    uint32_t magic_number;   /* magic number */
    uint16_t version_major;  /* major version number */
    uint16_t version_minor;  /* minor version number */
    int32_t  thiszone;       /* GMT to local correction */
    uint32_t sigfigs;        /* accuracy of timestamps */
    uint32_t snaplen;        /* max length of captured packets, in octets */
    uint32_t network;        /* data link type */
  };

  template< typename STREAM > inline void bwrite( STREAM & out, pcap_hdr_t const & o ) { 
    bwrite( out, o.magic_number );
    bwrite( out, o.version_major );
    bwrite( out, o.version_minor );
    bwrite( out, o.thiszone );
    bwrite( out, o.sigfigs );
    bwrite( out, o.snaplen );
    bwrite( out, o.network );
  }
  template< typename STREAM > inline void bread( STREAM & in, pcap_hdr_t & o ) { 
    bread( in, o.magic_number );
    bread( in, o.version_major );
    bread( in, o.version_minor );
    bread( in, o.thiszone );
    bread( in, o.sigfigs );
    bread( in, o.snaplen );
    bread( in, o.network );
  }

  struct pcaprec_hdr_t {
    // timestamp in seconds:microseconds
    uint32_t ts_sec;
    uint32_t ts_usec;
    // incl_len is # of bytes of packet saved in pcap file; orig_len is full original length. incl_len should be <=
    // orig_len. if it is (strictly) <, the packet was truncated.
    uint32_t incl_len; 
    uint32_t orig_len;
  };

  struct ip_frag_id_t {
    uint32_t ip_src;
    uint32_t ip_dst;
    uint16_t frag_id;
    uint8_t ip_p;
  };

  template< typename STREAM > inline void bwrite( STREAM & out, pcaprec_hdr_t const & o ) { 
    bwrite( out, o.ts_sec );
    bwrite( out, o.ts_usec );
    bwrite( out, o.incl_len );
    bwrite( out, o.orig_len );
  }
  template< typename STREAM > inline void bread( STREAM & in, pcaprec_hdr_t & o ) { 
    bread( in, o.ts_sec );
    bread( in, o.ts_usec );
    bread( in, o.incl_len );
    bread( in, o.orig_len );
  }

  struct data_stream_pcap_t : virtual public nesi, public data_stream_file_t // NESI(
                              // help="parse pcap file and output one block per packet/block in the file",
                              // bases=["data_stream_file_t"], type_id="pcap-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t extract_udp_payload; //NESI(default="0",help="if 1, assume ethernet+ip+udp packets, and extract udp payload")
    p_uint32_t udp_dest_port; //NESI(help="if set and extract_udp_payload=1, keep only payloads for this dest port")

    pcap_hdr_t hdr;

    vect_uint8_t frag_data; // if non-empty, in a fragment
    ip_frag_id_t ip_frag_id; // if in a fragment, the signature of it

    virtual bool seek_to_block( uint64_t const & frame_ix ) {
      printf( "pcap-src: seek to frame_ix=%s\n", str(frame_ix).c_str() );
      if( frame_ix != 0 ) { return false; } // only support restart      
      mfsr.seek_to_byte( 0 ); // FIXME: reset tot_num_read?
      pcap_src_init();
      return true;
    }

    void pcap_src_init( void ) {
      bread( mfsr, hdr );
      if( hdr.magic_number != pcap_file_magic ) { printf( "error reading pcap file; expected pcap_file_magic=%s, got hdr.magic_number=%s\n",
                                                          str(pcap_file_magic).c_str(), str(hdr.magic_number).c_str() ); }
      printf( "PCAP header: version_major=%s version_minor=%s thiszone=%s sigfigs=%s snaplen=%s network=%s\n",
              str(hdr.version_major).c_str(), str(hdr.version_minor).c_str(), str(hdr.thiszone).c_str(), str(hdr.sigfigs).c_str(),
              str(hdr.snaplen).c_str(), str(hdr.network).c_str() );
    }
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      data_stream_file_t::data_stream_init( nia );
      pcap_src_init();
    }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      while( 1 ) {
        if( mfsr.at_eof() ) { return ret; }
        pcaprec_hdr_t rec_hdr;
        bread( mfsr, rec_hdr );
        if( extract_udp_payload ) {
          ethernet_header const * eth_hdr = (ethernet_header const *)mfsr.consume_borrowed_raw_block( sizeof( ethernet_header ) ).get();
          uint16_t const ethertype = ntohs(eth_hdr->ethertype);
          if( ethertype != ethertype_ipv4 ) { rt_err( strprintf( "expected IPv4 ethertype but got ethertype=%s\n", str(ethertype).c_str() ) ); }
          ip const * ip_hdr = (ip const *)mfsr.consume_borrowed_raw_block( sizeof( ip ) ).get(); // with no options
          if( ip_hdr->ip_v != 4 ) { rt_err( strprintf( "error, expected IPv4 packet, but ip_hdr->ip_v=%s\n", str(ip_hdr->ip_v).c_str() ) ); }
          int32_t opts_sz = (int32_t(ip_hdr->ip_hl) << 2) - sizeof(ip);
          if( opts_sz < 0 ) { rt_err( strprintf( "error, bad IPv4 packet length ip_hdr->ip_hl=%s\n", str(ip_hdr->ip_hl).c_str() ) ); }
          mfsr.consume_and_discard_bytes( opts_sz );
          uint16_t const ip_id = ntohs(ip_hdr->ip_id);
          uint16_t const ip_off = ntohs(ip_hdr->ip_off);
          uint16_t const frag_off = (ip_off & IP_OFFMASK) << 3;
          //bool const ip_df = (ip_off & IP_DF);
          bool const ip_mf = (ip_off & IP_MF);
          uint32_t ip_payload_sz = rec_hdr.incl_len - (sizeof( ethernet_header ) + sizeof( ip ) + opts_sz);

          // we handle limited defragmenting here, assuming fragments are in order and there's never more than one
          // pending. however, we check this and should cleanly fail with an error if this assumption does not hold.
          if( ip_mf || (!frag_data.empty()) ) { // if a fragment, add it to fragment buffer
            if( frag_data.empty() ) { // if no fragment data, assume start of a fragment
              ip_frag_id.ip_src = ip_hdr->ip_src.s_addr;
              ip_frag_id.ip_dst = ip_hdr->ip_dst.s_addr;
              ip_frag_id.frag_id = ip_id;
              ip_frag_id.ip_p = ip_hdr->ip_p;
            } else {
              // fragment must match existing id and be the next part
              if( ip_frag_id.ip_src != ip_hdr->ip_src.s_addr ||
                  ip_frag_id.ip_dst != ip_hdr->ip_dst.s_addr ||
                  ip_frag_id.frag_id != ip_id ||
                  ip_frag_id.ip_p != ip_hdr->ip_p ) {
                // FIXME: yeah, even for pretty simple cases, it seems we need to handle this, sigh.
                printf( "warning: dropping error/unhandled: ip fragment if doesn't match id of in-progress partial packet being reassembled: ip_hdr->ip_src=%s ip_hdr->ip_dst=%s ip_hdr->ip_id=%s ip_hdr->ip_p=%s --- but currently --- ip_frag_id.ip_src=%s ip_frag_id.ip_dst=%s ip_frag_id.frag_id=%s ip_frag_id.ip_p=%s\n", str(ip_hdr->ip_src.s_addr).c_str(), str(ip_hdr->ip_dst.s_addr).c_str(), str(ip_id).c_str(), str(uint16_t(ip_hdr->ip_p)).c_str(), str(ip_frag_id.ip_src).c_str(), str(ip_frag_id.ip_dst).c_str(), str(ip_frag_id.frag_id).c_str(), str(uint16_t(ip_frag_id.ip_p)).c_str() );
                mfsr.consume_and_discard_bytes( ip_payload_sz );
                continue;
              }
              if( frag_off != frag_data.size() ) {
                rt_err( strprintf( "error/unhandled, ip fragment not contiguous. have %s bytes, but offset of next fragment is %s (should be ==)", str(frag_data.size()).c_str(), str(frag_off).c_str() ) );
              }
            }
            p_uint8_t pkt_frag_data = mfsr.consume_borrowed_raw_block( ip_payload_sz );
            frag_data.insert( frag_data.end(), pkt_frag_data.get(), pkt_frag_data.get() + ip_payload_sz );
            if( ip_mf ) { continue; } // continue unless this was the end of a fragment
          }
          // unfragmented packet OR end of a fragment
          printf( "frag_data.size()=%s\n", str(frag_data.size()).c_str() );
          printf( "ip_hdr->ip_p=%s\n", str(uint32_t(ip_hdr->ip_p)).c_str() );
          if( ip_hdr->ip_p != IPPROTO_UDP ) {
            // FIXME: dedupe this!
            if( frag_data.empty() ) {
              mfsr.consume_and_discard_bytes( ip_payload_sz );
            } else {
              frag_data.clear();
            }
            continue;
          }
          udphdr const * udp_hdr = 0;
          if( frag_data.empty() ) {
            udp_hdr = (udphdr const *)mfsr.consume_borrowed_raw_block( sizeof( udphdr ) ).get();
          } else {
            udp_hdr = (udphdr const *)&frag_data[0];
          }
          printf( "ntohs(udp_hdr->dest)=%s ntohs(udp_hdr->len)=%s\n", str(ntohs(udp_hdr->dest)).c_str(), str(ntohs(udp_hdr->len)).c_str() );
          int32_t udp_payload_sz =  ntohs(udp_hdr->len) - sizeof(udp_hdr);
          if( udp_payload_sz < 0 ) { rt_err( strprintf( "error, bad UDP packet length ntohs(upd_hdr.len)=%s\n", str(ntohs(udp_hdr->len)).c_str() ) ); }

          uint32_t expected_packet_size = 0;
          if( frag_data.empty() ) {
            expected_packet_size = sizeof( ethernet_header ) + sizeof( ip ) + opts_sz + sizeof( udphdr ) + udp_payload_sz;
            if( rec_hdr.incl_len != expected_packet_size ) {
              rt_err( strprintf( "rec_hdr.incl_len=%s rec_hdr.orig_len=%s, but expected_packet_size=%s (if == orig_len but incl_len is smaller, packet was truncated in capture. otherwise, if orig_len == incl_len, the stream is correct/unsupported)\n", str(rec_hdr.incl_len).c_str(), str(rec_hdr.orig_len).c_str(), str(expected_packet_size).c_str() ) );
            }
          } else {
            expected_packet_size = sizeof( udphdr ) + udp_payload_sz;
            if( frag_data.size() != expected_packet_size ) {
              rt_err( strprintf( "for reassembled UDP packet data, frag_data.size()=%s, but expected_packet_size=%s (UDP header and payload only)\n", str(frag_data.size()).c_str(), str(expected_packet_size).c_str() ) );
            }
          }
          if( udp_dest_port && (ntohs(udp_hdr->dest) != *udp_dest_port) ) {
            if( frag_data.empty() ) {
              mfsr.consume_and_discard_bytes( udp_payload_sz );
            } else {
              frag_data.clear();
            }
            continue;
          } else {
            if( frag_data.empty() ) {
              ret.nda = mfsr.consume_borrowed_block( udp_payload_sz );
            } else {
              ret.nda = make_shared<nda_t>( dims_t{ vect_uint32_t{uint32_t(udp_payload_sz)}, vect_string{ "v" }, "uint8_t" } );
              std::copy( frag_data.begin(), frag_data.end(), (uint8_t *)ret.d() );
              frag_data.clear();
            }
          }
        } else {
          // FIXME: for now, we just pass along the incl_len part, and we discard orig_len ...
          ret.nda = mfsr.consume_borrowed_block( rec_hdr.incl_len );
        }
        ret.timestamp_ns = (uint64_t(rec_hdr.ts_sec)*1000*1000+uint64_t(rec_hdr.ts_usec))*1000;
        return ret;
      }
    }
  };
  
  struct data_sink_pcap_t : virtual public nesi, public data_sink_file_t // NESI(
                              // help="parse pcap file and output one block per packet/block in the file",
                              // bases=["data_sink_file_t"], type_id="pcap-sink")
  {
    uint32_t add_header; //NESI(default="0",help="if 1, add some zero header bytes from the start of each block")
    uint32_t add_header_bytes; //NESI(default="42",help="if add_header=1, number of (zero) bytes to add to the start of each block. default is 42 (ethernet), but note that the actual pcap network type doesn't affect this and isn't checked/used.")
    string header_smac; //NESI(default="ffffffffffff",help="(FIXME: not impl yet) if add_header=1, use this src mac in generated header")
    string header_dmac; //NESI(default="010203040506",help="(FIXME: not impl yet) if add_header=1, use this dest mac in generated header")

    uint32_t header_upd_sport; //NESI(default="443",help="if add_header=1, use this udp src port in generated header")
    uint32_t header_upd_dport; //NESI(default="2368",help="if add_header=1, use this udp dest port in generated header")
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    pcap_hdr_t hdr;

    vect_uint8_t header;

    virtual string get_pos_info_str( void ) { return string("data_sink_pcal: wrote <NOT_IMPL> packets to:") + data_sink_file_t::get_pos_info_str(); }
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      data_sink_file_t::data_stream_init( nia );
      hdr.magic_number = pcap_file_magic;
      hdr.version_major = 2;
      hdr.version_minor = 4;
      hdr.thiszone = 0;
      hdr.sigfigs = 0;
      hdr.snaplen = 1 << 18; // note: pretty arbitrary, maybe not meaningful ...
      hdr.network = 1;
      bwrite( *out, hdr );
      header.resize( add_header_bytes, 0 ); // could fill this in i suppose ...
      header_smac = unhex( header_smac );
      header_dmac = unhex( header_dmac );
      if( header_smac.size() != 6 ) { rt_err( "src mac should be 6 bytes as hex with no spaces/seperators" ); }
      if( header_dmac.size() != 6 ) { rt_err( "dest mac should be 6 bytes as hex with no spaces/seperators" ); }
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      if( !db.nda ) { rt_err( "pcap-sink: expected data block to have data, but db.nda was null."); }
      pcaprec_hdr_t rec_hdr;
      uint64_t timestamp_us = db.timestamp_ns / 1000;
      rec_hdr.ts_sec = timestamp_us / ( 1000 * 1000 );
      rec_hdr.ts_usec = timestamp_us % ( 1000 * 1000 );
      uint64_t rec_len = db.sz();
      if( add_header ) { rec_len += add_header_bytes; }
      rec_hdr.incl_len = rec_len;
      rec_hdr.orig_len = rec_len; // FIXME: as per comment in reader, we don't know this here ...
      bwrite( *out, rec_hdr );
      if( add_header ) { // add fake eth/ip/udp header (14 +  + = 42 bytes total)
        //bwrite_bytes( *out, (char const *)&header[0], header.size() );
        ethernet_header eth_hdr = {};
        // FIXME: not sure about byte order here, and doesn't seem needed ...
        //std::copy( header_smac.begin(), header_smac.end(), eth_hdr.src_mac );
        //std::copy( header_dmac.begin(), header_dmac.end(), eth_hdr.dest_mac );
        eth_hdr.ethertype = htons(ethertype_ipv4); 
        bwrite_bytes( *out, (char const *)&eth_hdr, sizeof(eth_hdr) );
        ip ip_hdr = {};
        ip_hdr.ip_v = 4;
        assert_st( !( sizeof(ip_hdr) & 0x3 ) );
        ip_hdr.ip_hl = sizeof(ip_hdr) >> 2;
        ip_hdr.ip_p = 17;
        ip_hdr.ip_len = htons( uint16_t( sizeof(ip) + sizeof(udphdr) + db.sz() ));
        ip_hdr.ip_id = htons( uint16_t( 1 ) );
        ip_hdr.ip_ttl = 128;
        ip_hdr.ip_sum = in_cksum( (uint16_t *)&ip_hdr, sizeof(ip_hdr) >> 1 );
        bwrite_bytes( *out, (char const *)&ip_hdr, sizeof(ip_hdr) );
        udphdr udp_hdr = {};
        udp_hdr.source = htons( uint16_t( header_upd_sport ) );
        udp_hdr.dest = htons( uint16_t( header_upd_dport ) );
        udp_hdr.len = htons( uint16_t( sizeof(udp_hdr) + db.sz() ) );
        bwrite_bytes( *out, (char const *)&udp_hdr, sizeof(udp_hdr) );
      }
      bwrite_bytes( *out, (char const *)db.d(), db.sz() );
      return db;
    }
    
  };

#include"gen/data-stream-pcap.cc.nesi_gen.cc"

}
