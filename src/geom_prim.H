// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef GEOM_PRIM_H
#define GEOM_PRIM_H
#include"boda_base.H"
#include<cstdint>

namespace boda 
{
  template< typename T > struct T_pt_t {
    typedef T value_type;
    T d[2];
    T_pt_t( void ) : d{0,0} { } // no c++11? --> remove : d{0,0}, use { d[0]=0;d[1]=0; }
    T_pt_t( T const d0, T const d1 ) : d{d0,d1} { } 
    T dims_prod( void ) const { return d[0] * d[1]; }
    T dims_min( void ) const { return std::min(d[0],d[1]); }
    T dims_max( void ) const { return std::max(d[0],d[1]); }
    void read_from_line_parts( vect_string const & parts, uint32_t const init_ix );
    T_pt_t scale_and_round( double const & s ) const { return T_pt_t( round(d[0]*s), round(d[1]*s) ); }
    T_pt_t scale_and_round( T_pt_t<double> const & s ) const { return T_pt_t( round(d[0]*s.d[0]), round(d[1]*s.d[1]) ); }
    T_pt_t scale( T const & o ) const { return T_pt_t( d[0] * o, d[1] * o ); }

    void max_eq( T_pt_t const & o ) { boda::max_eq(d[0],o.d[0]); boda::max_eq(d[1],o.d[1]); }
    void min_eq( T_pt_t const & o ) { boda::min_eq(d[0],o.d[0]); boda::min_eq(d[1],o.d[1]); }
    void operator += ( T_pt_t const & o ) { d[0] += o.d[0]; d[1] += o.d[1]; }
    void operator -= ( T_pt_t const & o ) { d[0] -= o.d[0]; d[1] -= o.d[1]; }

    T_pt_t operator + ( T_pt_t const & o ) const { return T_pt_t( d[0] + o.d[0], d[1] + o.d[1] ); }
    T_pt_t operator - ( T_pt_t const & o ) const { return T_pt_t( d[0] - o.d[0], d[1] - o.d[1] ); }
    T_pt_t sub_sat_zero( T_pt_t const & o ) const { return T_pt_t( (d[0]>o.d[0])?(d[0]-o.d[0]):0,
								   (d[1]>o.d[1])?(d[1]-o.d[1]):0 ); }
    T_pt_t operator * ( T_pt_t const & o ) const { return T_pt_t( d[0] * o.d[0], d[1] * o.d[1] ); }
    T_pt_t operator / ( T_pt_t const & o ) const { return T_pt_t( d[0] / o.d[0], d[1] / o.d[1] ); }
    T_pt_t operator << ( uint32_t const & o ) const { return T_pt_t( d[0] << o, d[1] << o ); }
    T_pt_t operator >> ( uint32_t const & o ) const { return T_pt_t( d[0] >> o, d[1] >> o ); }
    bool is_zeros( void ) const { return (!d[0])&&(!d[1]); }
    bool dims_are_same( void ) const { return d[0]==d[1]; }
    bool both_dims_non_zero( void ) const { return d[0]&&d[1]; }
    bool both_dims_ge_zero( void ) const { return (d[0]>=0)&&(d[1]>=0); }
    bool operator == ( T_pt_t const & o ) const { return d[0] == o.d[0] && d[1] == o.d[1]; }
    bool operator != ( T_pt_t const & o ) const { return !(*this == o); }
    bool operator <  ( T_pt_t const & o ) const { return (d[0] != o.d[0]) ? (d[0] < o.d[0]) : (d[1] < o.d[1]); }
    bool both_dims_lt( T_pt_t const & o ) const { return d[0] <  o.d[0] && d[1] <  o.d[1]; }
    bool both_dims_gt( T_pt_t const & o ) const { return d[0] >  o.d[0] && d[1] >  o.d[1]; }
    bool both_dims_le( T_pt_t const & o ) const { return d[0] <= o.d[0] && d[1] <= o.d[1]; }
    bool both_dims_ge( T_pt_t const & o ) const { return d[0] >= o.d[0] && d[1] >= o.d[1]; }
    bool bnds_check_gele( T const & o ) const { return (o >= d[0]) && (o <= d[1]); }
  };

  template< typename T > inline T_pt_t<T> min( T_pt_t<T> const &a, T_pt_t<T> const & b ) { 
    return {std::min(a.d[0],b.d[0]),std::min(a.d[1],b.d[1])}; }
  template< typename T > inline T_pt_t<T> max( T_pt_t<T> const &a, T_pt_t<T> const & b ) { 
    return {std::max(a.d[0],b.d[0]),std::max(a.d[1],b.d[1])}; }

  typedef T_pt_t< uint32_t > u32_pt_t;
  typedef T_pt_t< int32_t  > i32_pt_t;
  typedef T_pt_t< double  > double_pt_t;


  inline u32_pt_t floor_div_u32( u32_pt_t const &v, uint32_t const & d ) { return u32_pt_t( v.d[0] / d, v.d[1] / d ); }

  inline i32_pt_t u32_to_i32( u32_pt_t const & v ) {
    i32_pt_t const ret( v.d[0], v.d[1] );
    assert( ret.both_dims_ge_zero() ); // overflow check (hopefully: assumes unrep uint will wrap to neg)
    return ret;
  }
  inline u32_pt_t i32_to_u32( i32_pt_t const & v ) { assert( v.both_dims_ge_zero() ); return u32_pt_t( v.d[0], v.d[1] ); }

  inline u32_pt_t ceil_div( u32_pt_t const &v, u32_pt_t const & d ) { 
    return u32_pt_t( u32_ceil_div(v.d[0],d.d[0]), u32_ceil_div(v.d[1],d.d[1]) ); }
  inline u32_pt_t ceil_align( u32_pt_t const &v, u32_pt_t const & align ) { return ceil_div( v, align ) * align; }

  inline i32_pt_t floor_div( i32_pt_t const &v, u32_pt_t const & d ) { 
    i32_pt_t const id = u32_to_i32( d );
    return i32_pt_t( i32_floor_div(v.d[0],id.d[0]), i32_floor_div(v.d[1],id.d[1]) ); 
  }
  inline i32_pt_t ceil_div( i32_pt_t const &v, u32_pt_t const & d ) { 
    i32_pt_t const id = u32_to_i32( d );
    return i32_pt_t( i32_ceil_div(v.d[0],id.d[0]), i32_ceil_div(v.d[1],id.d[1]) ); 
  }

  std::ostream & operator <<(std::ostream & os, u32_pt_t const & v);
  std::ostream & operator <<(std::ostream & os, i32_pt_t const & v);

  u32_pt_t const u32_pt_t_const_max{uint32_t_const_max,uint32_t_const_max};
  
  typedef vector< u32_pt_t > vect_u32_pt_t; 
  typedef vector< vect_u32_pt_t > vect_vect_u32_pt_t; 
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

  template< typename T > struct T_box_t {
    typedef T value_type;
    T p[2];
    T_box_t( void ) { }
    T_box_t( T const & p0, T const & p1 ) : p{p0,p1} { } 
    bool is_zeros( void ) const { return p[0].is_zeros()&&p[1].is_zeros(); }
    bool is_normalized( void ) const { return (p[0].d[0]<=p[1].d[0]) && (p[0].d[1]<=p[1].d[1]); }
    bool is_strictly_normalized( void ) const { return (p[0].d[0]<p[1].d[0]) && (p[0].d[1]<p[1].d[1]); }
    void normalize( void ) { // for both dims, if lb>ub, swap them. afterwards, this->is_normalized() will hold
      if( p[0].d[0]>p[1].d[0] ) { std::swap( p[0].d[0], p[1].d[0] ); }
      if( p[0].d[1]>p[1].d[1] ) { std::swap( p[0].d[1], p[1].d[1] ); }
    }
    // all overlap functions assume both this->is_strictly_normalized() && o.is_strictly_normalized() 
    // (otherwise, they do what they do, but the names don't mean as much)
    void get_overlap_bnds( T & lb, T & ub, T_box_t const & o ) const {
      lb.d[0] = std::max( p[0].d[0], o.p[0].d[0] );
      ub.d[0] = std::min( p[1].d[0], o.p[1].d[0] );
      lb.d[1] = std::max( p[0].d[1], o.p[0].d[1] );
      ub.d[1] = std::min( p[1].d[1], o.p[1].d[1] );
    }
    uint64_t get_overlap_with( T_box_t const & o ) const {
      T lb, ub;
      get_overlap_bnds( lb, ub, o );
      if( ( lb.d[0] >= ub.d[0] ) || ( lb.d[1] >= ub.d[1] ) ) { return 0; }
      return uint64_t(ub.d[0]-lb.d[0])*(ub.d[1]-lb.d[1]);
    }
    bool has_overlap_with( T_box_t const & o ) const {
      T lb, ub;
      get_overlap_bnds( lb, ub, o );
      return !( ( lb.d[0] >= ub.d[0] ) || ( lb.d[1] >= ub.d[1] ) );
    }
    T_box_t get_overlap_box_with( T_box_t const & o ) const {
      T_box_t ret;
      get_overlap_bnds( ret.p[0], ret.p[1], o );
      return ret; //note: if !ret.is_strictly_normalized(), there was no overlap (assuming o/this is_strictly_normalized())
    }
    bool contains( T_box_t const & o ) const { return o.p[0].both_dims_ge( p[0] ) && o.p[1].both_dims_le( p[1] ); }
    bool strictly_contains( T const & o ) const { return o.both_dims_ge( p[0] ) && o.both_dims_lt( p[1] ); }
    uint64_t get_area( void ) const { return uint64_t(p[1].d[0]-p[0].d[0])*(p[1].d[1]-p[0].d[1]); } 
    string parts_str( void ) { return str( p[0] ) + " " + str( p[1] ); }
    void read_from_line_parts( vect_string const & parts, uint32_t const init_ix );
    void from_pascal_coord_adjust( void );
    void to_pascal_coord_adjust( void );
    std::string pascal_str( void ) const;
    T_box_t operator + ( T_box_t const & o ) const { return T_box_t( p[0] + o.p[0], p[1] + o.p[1] ); }
    T_box_t operator - ( T_box_t const & o ) const { return T_box_t( p[0] - o.p[0], p[1] - o.p[1] ); }
    void max_eq( T_box_t const & o ) { p[0].max_eq(o.p[0]); p[1].max_eq(o.p[1]); }
    void min_eq( T_box_t const & o ) { p[0].min_eq(o.p[0]); p[1].min_eq(o.p[1]); }
    void operator += ( T_box_t const & o ) { p[0] += o.p[0]; p[1] += o.p[1]; }
    void operator -= ( T_box_t const & o ) { p[0] -= o.p[0]; p[1] -= o.p[1]; }
    void operator += ( T const & o ) { p[0] += o; p[1] += o; }
    void operator -= ( T const & o ) { p[0] -= o; p[1] -= o; }
    T_box_t operator + ( T const & o ) const { return T_box_t( p[0] + o, p[1] + o ); }
    T_box_t operator - ( T const & o ) const { return T_box_t( p[0] - o, p[1] - o ); }
    T_box_t operator * ( T const & o ) const { return T_box_t( p[0] * o, p[1] * o ); }
    T_box_t operator / ( T const & o ) const { return T_box_t( p[0] / o, p[1] / o ); }
    bool operator == ( T_box_t const & o ) const { return p[0] == o.p[0] && p[1] == o.p[1]; }
    bool operator != ( T_box_t const & o ) const { return !(*this == o); }
    bool operator <  ( T_box_t const & o ) const { return (p[0] != o.p[0]) ? (p[0] < o.p[0]) : (p[1] < o.p[1]); }
    T sz( void ) const { return p[1] - p[0]; }
    T bnds_sum( void ) const { return p[0] + p[1]; }
    bool bnds_are_same( void ) const { return p[0]==p[1]; }
    T_box_t scale_dims( T const & scale ) const { return T_box_t( p[0] * scale, p[1] * scale ); }    
    T_box_t scale( typename T::value_type const & s ) const { return T_box_t( p[0].scale( s ), p[1].scale( s ) ); }    
    T_box_t scale_and_round( double const & s ) const { return T_box_t( p[0].scale_and_round(s), p[1].scale_and_round(s) ); }
    T_box_t scale_and_round( double_pt_t const & s ) const { return T_box_t( p[0].scale_and_round(s), p[1].scale_and_round(s) ); }

    T center_ru( void ) const { return ceil_div( bnds_sum(), u32_pt_t{2,2} ); }
    T center_rd( void ) const { return floor_div( bnds_sum(), u32_pt_t{2,2} ); }
    T dim_wise_max( void ) const { return T( std::max(p[0].d[0],p[1].d[0]), std::max(p[0].d[1],p[1].d[1]) ); }
  };

  template< typename T > T_box_t<T> pel_to_box( T const & pel ) { return T_box_t<T>( pel, pel + T{1,1} ); }

  typedef T_box_t< u32_pt_t > u32_box_t;
  typedef T_box_t< i32_pt_t > i32_box_t;

  // compare two points by product (i.e. area if pt's are sizes of boxes) >
  struct u32_pt_t_by_prod_gt_t {
    bool operator () ( u32_pt_t const & a, u32_pt_t const & b ) const { return a.dims_prod() > b.dims_prod(); }
  };
  inline i32_box_t u32_to_i32( u32_box_t const & v ) { return i32_box_t( u32_to_i32( v.p[0] ), u32_to_i32( v.p[1] ) ); }
  inline u32_box_t i32_to_u32( i32_box_t const & v ) { return u32_box_t( i32_to_u32(v.p[0]), i32_to_u32(v.p[1] ) ); }

  inline i32_box_t floor_div( i32_box_t const &v, u32_pt_t const & d ) { 
    return i32_box_t( floor_div( v.p[0], d ), floor_div( v.p[1], d ) ); }
  inline i32_box_t ceil_div( i32_box_t const &v, u32_pt_t const & d ) { 
    return i32_box_t( ceil_div( v.p[0], d ), ceil_div( v.p[1], d ) ); }
    
  typedef set< i32_pt_t > set_i32_pt_t;
  typedef map< i32_pt_t, uint32_t > map_i32_pt_t_to_uint32_t;
  typedef map< i32_pt_t, i32_box_t > map_i32_pt_t_to_i32_box_t;
  typedef set< u32_pt_t > set_u32_pt_t;

  typedef vector< u32_box_t > vect_u32_box_t; 
  typedef set< u32_box_t > set_u32_box_t;
  typedef set< i32_box_t > set_i32_box_t;
  typedef shared_ptr< u32_box_t > p_u32_box_t; 
  typedef vector< p_u32_box_t > vect_p_u32_box_t;
  typedef shared_ptr< vect_u32_box_t > p_vect_u32_box_t; 


  std::ostream & operator <<(std::ostream & os, u32_box_t const & v);
  std::ostream & operator <<(std::ostream & os, i32_box_t const & v);

  // misc higher-level operations
  uint32_t blf_place( vect_u32_pt_w_t & out, u32_box_t const & bin, vect_u32_pt_t const & to_place, bool const no_fit_okay );

};
#endif // GEOM_PRIM_H
