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
  // from libpcap documentation, we learned the pcap file format global and packet header format: https://wiki.wireshark.org/Development/LibpcapFileFormat
  uint32_t const pcap_file_magic = 0xa1b2c3d4;

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
                              // bases=["data_stream_file_t"], type_id="pcap")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t drop_header; //NESI(default="0",help="if 1, drop some header bytes from the start of each block")
    uint32_t drop_header_bytes; //NESI(default="42",help="if drop_header=1, drop number of bytes to drop from the start of each block. default is 42 (ethernet), but note that the actual pcap network type doesn't affect this and isn't checked/used.")

    pcap_hdr_t hdr;
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      data_stream_file_t::data_stream_init( nia );
      bread( mfsr, hdr );
      if( hdr.magic_number != pcap_file_magic ) { printf( "error reading pcap file; expected pcap_file_magic=%s, got hdr.magic_number=%s\n",
                                                          str(pcap_file_magic).c_str(), str(hdr.magic_number).c_str() ); }
      printf( "PCAP header: version_major=%s version_minor=%s thiszone=%s sigfigs=%s snaplen=%s network=%s\n",
              str(hdr.version_major).c_str(), str(hdr.version_minor).c_str(), str(hdr.thiszone).c_str(), str(hdr.sigfigs).c_str(),
              str(hdr.snaplen).c_str(), str(hdr.network).c_str() );
    }
    
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      if( mfsr.at_eof() ) { return ret; }
      pcaprec_hdr_t rec_hdr;
      bread( mfsr, rec_hdr );
      if( drop_header ) {
        if( rec_hdr.incl_len < drop_header_bytes ) {
          rt_err( strprintf( "error, can't drop drop_header_bytes=%s from packet with rec_hdr.incl_len=%s\n",
                             str(rec_hdr.incl_len).c_str(), str(drop_header_bytes).c_str() ) );
        }
        mfsr.consume_and_discard_bytes( drop_header_bytes );
        rec_hdr.incl_len -= drop_header_bytes;
      }
        
      // FIXME: for now, we just pass along the incl_len part, and we discard orig_len ...
      ret = mfsr.consume_borrowed_block( rec_hdr.incl_len );
      ret.timestamp_ns = (uint64_t(rec_hdr.ts_sec)*1000*1000+uint64_t(rec_hdr.ts_usec))*1000;

      if( verbose && (!drop_header) ) {
        uint8_t * hdr = ret.d.get();
        hdr += sizeof( ethernet_header );
        struct ip *ip_hdr = (struct ip *)hdr;
        printf( "ip_hdr->ip_v=%s ip_hdr->ip_p=%s\n", str(ip_hdr->ip_v).c_str(), str(uint32_t(ip_hdr->ip_p)).c_str() );
        hdr += ip_hdr->ip_hl << 2;
        struct udphdr *udp_hdr = (struct udphdr *)hdr;
        printf( "ntohs(udp_hdr->dest)=%s ntohs(udp_hdr->len)=%s\n", str(ntohs(udp_hdr->dest)).c_str(), str(ntohs(udp_hdr->len)).c_str() );        
      }
      return ret;
    }
  };
  unsigned short
  in_cksum(unsigned short *addr, int len)
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

  
  struct data_sink_pcap_t : virtual public nesi, public data_sink_file_t // NESI(
                              // help="parse pcap file and output one block per packet/block in the file",
                              // bases=["data_sink_file_t"], type_id="pcap")
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
    
    virtual void data_sink_init( nesi_init_arg_t * const nia ) {
      data_sink_file_t::data_sink_init( nia );
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

    virtual void consume_block( data_block_t const & db ) {
      pcaprec_hdr_t rec_hdr;
      uint64_t timestamp_us = db.timestamp_ns / 1000;
      rec_hdr.ts_sec = timestamp_us / ( 1000 * 1000 );
      rec_hdr.ts_usec = timestamp_us % ( 1000 * 1000 );
      uint64_t rec_len = db.sz;
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
        eth_hdr.ethertype = 0x08;
        bwrite_bytes( *out, (char const *)&eth_hdr, sizeof(eth_hdr) );
        ip ip_hdr = {};
        ip_hdr.ip_v = 4;
        assert_st( !( sizeof(ip_hdr) & 0x3 ) );
        ip_hdr.ip_hl = sizeof(ip_hdr) >> 2;
        ip_hdr.ip_p = 17;
        ip_hdr.ip_len = htons( uint16_t( sizeof(ip) + sizeof(udphdr) + db.sz ));
        ip_hdr.ip_id = htons( uint16_t( 1 ) );
        ip_hdr.ip_ttl = 128;
        ip_hdr.ip_sum = in_cksum( (uint16_t *)&ip_hdr, sizeof(ip_hdr) >> 1 );
        bwrite_bytes( *out, (char const *)&ip_hdr, sizeof(ip_hdr) );
        udphdr udp_hdr = {};
        udp_hdr.source = htons( uint16_t( header_upd_sport ) );
        udp_hdr.dest = htons( uint16_t( header_upd_dport ) );
        udp_hdr.len = htons( uint16_t( sizeof(udp_hdr) + db.sz ) );
        bwrite_bytes( *out, (char const *)&udp_hdr, sizeof(udp_hdr) );
      }
      bwrite_bytes( *out, (char const *)db.d.get(), db.sz );
    }
    
  };

#include"gen/data-stream-pcap.cc.nesi_gen.cc"

}
