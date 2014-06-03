// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef GEOM_PRIM_H
#define GEOM_PRIM_H
#include"boda_base.H"
#include<cstdint>

namespace boda 
{
  struct u32_pt_t
  {
    uint32_t d[2];
    u32_pt_t( void ) : d{0,0} { } // no c++11? --> remove : d{0,0}, use { d[0]=0;d[1]=0; }
    u32_pt_t( uint32_t const d0, uint32_t const d1 ) : d{d0,d1} { } 
    uint32_t dims_prod( void ) const { return d[0] * d[1]; }
    uint32_t dims_min( void ) const { return std::min(d[0],d[1]); }
    uint32_t dims_max( void ) const { return std::max(d[0],d[1]); }
    void read_from_line_parts( vect_string const & parts, uint32_t const init_ix );
    u32_pt_t scale_and_round( double const & s ) const { return u32_pt_t( round(d[0]*s), round(d[1]*s) ); }

    void operator += ( u32_pt_t const & o ) { d[0] += o.d[0]; d[1] += o.d[1]; }
    void operator -= ( u32_pt_t const & o ) { d[0] -= o.d[0]; d[1] -= o.d[1]; }

    u32_pt_t operator + ( u32_pt_t const & o ) const { return u32_pt_t( d[0] + o.d[0], d[1] + o.d[1] ); }
    u32_pt_t operator - ( u32_pt_t const & o ) const { return u32_pt_t( d[0] - o.d[0], d[1] - o.d[1] ); }
    u32_pt_t operator * ( u32_pt_t const & o ) const { return u32_pt_t( d[0] * o.d[0], d[1] * o.d[1] ); }
    u32_pt_t operator / ( u32_pt_t const & o ) const { return u32_pt_t( d[0] / o.d[0], d[1] / o.d[1] ); }
    u32_pt_t operator << ( uint32_t const & o ) const { return u32_pt_t( d[0] << o, d[1] << o ); }
    u32_pt_t operator >> ( uint32_t const & o ) const { return u32_pt_t( d[0] >> o, d[1] >> o ); }
    bool is_zeros( void ) const { return (!d[0])&&(!d[1]); }
    bool both_dims_non_zero( void ) const { return d[0]&&d[1]; }
    bool operator == ( u32_pt_t const & o ) const { return d[0] == o.d[0] && d[1] == o.d[1]; }
    bool operator != ( u32_pt_t const & o ) const { return !(*this == o); }
    bool operator <  ( u32_pt_t const & o ) const { return (d[0] != o.d[0]) ? (d[0] < o.d[0]) : (d[1] < o.d[1]); }
    bool both_dims_lt( u32_pt_t const & o ) const { return d[0] <  o.d[0] && d[1] <  o.d[1]; }
    bool both_dims_gt( u32_pt_t const & o ) const { return d[0] >  o.d[0] && d[1] >  o.d[1]; }
    bool both_dims_le( u32_pt_t const & o ) const { return d[0] <= o.d[0] && d[1] <= o.d[1]; }
    bool both_dims_ge( u32_pt_t const & o ) const { return d[0] >= o.d[0] && d[1] >= o.d[1]; }
    u32_pt_t ceil_align( u32_pt_t const & align ) const { 
      return u32_pt_t( u32_ciel_align(d[0],align.d[0]), u32_ciel_align(d[1],align.d[1]) ); 
    }
  };
  std::ostream & operator <<(std::ostream & os, u32_pt_t const & v);

  template< typename HAS_WH > u32_pt_t get_wh( HAS_WH const & o ) { return u32_pt_t( o.w, o.h ); }

  u32_pt_t const u32_pt_t_const_max{uint32_t_const_max,uint32_t_const_max};
  
  typedef vector< u32_pt_t > vect_u32_pt_t; 
  typedef shared_ptr< u32_pt_t > p_u32_pt_t; 
  typedef vector< p_u32_pt_t > vect_p_u32_pt_t;
  typedef shared_ptr< vect_u32_pt_t > p_vect_u32_pt_t; 

  // note: T must be a type initializable with 0
  template< typename T >
  struct u32_pt_T_t : public u32_pt_t {
    T w;
    u32_pt_T_t( void ) : w(0) { }
    u32_pt_T_t( uint32_t const d0, uint32_t const d1 ) : u32_pt_t(d0,d1), w(0) { } 
    u32_pt_T_t( uint32_t const d0, uint32_t const d1, uint32_t const w_ ) : u32_pt_t(d0,d1), w(w_) { } 
    u32_pt_T_t( u32_pt_t const d_, T const & w_ ) : u32_pt_t(d_), w(w_) { } 
  };
  typedef u32_pt_T_t< uint32_t > u32_pt_w_t;
  std::ostream & operator <<(std::ostream & os, u32_pt_w_t const & v);
  typedef vector< u32_pt_w_t > vect_u32_pt_w_t; 
  typedef shared_ptr< vect_u32_pt_w_t > p_vect_u32_pt_w_t; 
  typedef shared_ptr< u32_pt_w_t > p_u32_pt_w_t; 
  typedef vector< p_u32_pt_w_t > vect_p_u32_pt_w_t;

  typedef u32_pt_T_t< float > u32_pt_f_t;
  std::ostream & operator <<(std::ostream & os, u32_pt_f_t const & v);
  typedef vector< u32_pt_f_t > vect_u32_pt_f_t; 
  typedef shared_ptr< vect_u32_pt_f_t > p_vect_u32_pt_f_t; 
  typedef shared_ptr< u32_pt_f_t > p_u32_pt_f_t; 
  typedef vector< p_u32_pt_f_t > vect_p_u32_pt_f_t;

  struct u32_box_t
  {
    u32_pt_t p[2];
    u32_box_t( void ) { }
    u32_box_t( u32_pt_t const & p0, u32_pt_t const & p1 ) : p{p0,p1} { } 
    bool is_normalized( void ) const { return (p[0].d[0]<=p[1].d[0]) && (p[0].d[1]<=p[1].d[1]); }
    bool is_strictly_normalized( void ) const { return (p[0].d[0]<p[1].d[0]) && (p[0].d[1]<p[1].d[1]); }
    void normalize( void ) { // for both dims, if lb>ub, swap them. afterwards, this->is_normalized() will hold
      if( p[0].d[0]>p[1].d[0] ) { std::swap( p[0].d[0], p[1].d[0] ); }
      if( p[0].d[1]>p[1].d[1] ) { std::swap( p[0].d[1], p[1].d[1] ); }
    }
    // all overlap functions assume both this->is_strictly_normalized() && o.is_strictly_normalized() 
    // (otherwise, they do what they do, but the names don't mean as much)
    void get_overlap_bnds( uint32_t & lb0, uint32_t & ub0, uint32_t & lb1, uint32_t & ub1, u32_box_t const & o ) const {
      lb0 = std::max( p[0].d[0], o.p[0].d[0] );
      ub0 = std::min( p[1].d[0], o.p[1].d[0] );
      lb1 = std::max( p[0].d[1], o.p[0].d[1] );
      ub1 = std::min( p[1].d[1], o.p[1].d[1] );
    }
    uint64_t get_overlap_with( u32_box_t const & o ) const {
      uint32_t lb0, ub0, lb1, ub1;
      get_overlap_bnds( lb0, ub0, lb1, ub1, o );
      if( ( lb0 >= ub0 ) || ( lb1 >= ub1 ) ) { return 0; }
      return uint64_t(ub0-lb0)*(ub1-lb1);
    }
    bool has_overlap_with( u32_box_t const & o ) const {
      uint32_t lb0, ub0, lb1, ub1;
      get_overlap_bnds( lb0, ub0, lb1, ub1, o );
      return !( ( lb0 >= ub0 ) || ( lb1 >= ub1 ) );
    }
    u32_box_t get_overlap_box_with( u32_box_t const & o ) const {
      u32_box_t ret;
      get_overlap_bnds( ret.p[0].d[0], ret.p[1].d[0], ret.p[0].d[1], ret.p[1].d[1], o );
      return ret; // note: if !ret.is_strictly_normalized(), there was no overlap (assuming o/this is_strictly_normalized() )
    }

    uint64_t get_area( void ) const { return uint64_t(p[1].d[0]-p[0].d[0])*(p[1].d[1]-p[0].d[1]); } 

    void read_from_line_parts( vect_string const & parts, uint32_t const init_ix );
    void from_pascal_coord_adjust( void );
    void to_pascal_coord_adjust( void );
    std::string pascal_str( void ) const;
    u32_box_t operator + ( u32_box_t const & o ) const { return u32_box_t( p[0] + o.p[0], p[1] + o.p[1] ); }
    bool operator == ( u32_box_t const & o ) const { return p[0] == o.p[0] && p[1] == o.p[1]; }
    bool operator != ( u32_box_t const & o ) const { return !(*this == o); }
    bool operator <  ( u32_box_t const & o ) const { return (p[0] != o.p[0]) ? (p[0] < o.p[0]) : (p[1] < o.p[1]); }
    u32_pt_t sz( void ) const { return p[1] - p[0]; }
    u32_pt_t bnds_sum( void ) const { return p[0] + p[1]; }
    u32_box_t scale_dims( u32_pt_t const & scale ) const { return u32_box_t( p[0] * scale, p[1] * scale ); }
    
  };

  // compare two points by product (i.e. area if pt's are sizes of boxes) >
  struct u32_pt_t_by_prod_gt_t {
    bool operator () ( u32_pt_t const & a, u32_pt_t const & b ) const { return a.dims_prod() > b.dims_prod(); }
  };
    
  typedef vector< u32_box_t > vect_u32_box_t; 
  typedef set< u32_box_t > set_u32_box_t;
  typedef shared_ptr< u32_box_t > p_u32_box_t; 
  typedef vector< p_u32_box_t > vect_p_u32_box_t;
  typedef shared_ptr< vect_u32_box_t > p_vect_u32_box_t; 


  std::ostream & operator <<(std::ostream & os, u32_box_t const & v);

  // misc higher-level operations
  uint32_t blf_place( vect_u32_pt_w_t & out, u32_box_t const & bin, vect_u32_pt_t const & to_place, bool const no_fit_okay );

};
#endif // GEOM_PRIM_H
