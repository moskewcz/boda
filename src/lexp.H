// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _LEXP_H_
#define _LEXP_H_

#include"boda_base.H"

namespace pugi { class xml_node; }

namespace boda {

// sub-string pointer type with begin/end. references portion of larger shared string. uses p_uint8_t for base storage.
  struct sstr_t
  {
    p_uint8_t base;
    uint32_t b;
    uint32_t e;
    sstr_t( void ) : b(0), e(0) { }
    void set_from_string( std::string const &s );
    void set_from_cstr( char const * s );
    void borrow_from_string( std::string const &s );
    void borrow_from_cstr( char const * s );
    std::string str( void ) const { 
      assert(base); return std::string( base.get()+b, base.get()+e ); } // get copy as std::string
    uint32_t sz( void ) const { assert(base); assert( e >= b ); return e - b; }
    uint32_t end_off( void ) const { assert(base); return e; } 
    bool exists( void ) const { return base != 0; }
    void shrink_end_off( uint32_t const new_e )
    {
      assert( new_e >= b ); // must yield sz >= 0
      assert( new_e <= e ); // must yield new_sz <= sz
      e = new_e;
    }
    bool operator < ( sstr_t const & o ) const;
  };
  std::ostream & operator <<(std::ostream & os, sstr_t const & v);
  struct lexp_t;
  struct lexp_stats_t; // for testing/internal use only
  typedef shared_ptr< lexp_t > p_lexp_t;
// name/value pair
  struct lexp_nv_t
  {
    sstr_t n;
    p_lexp_t v;
  };
  std::ostream & operator <<(std::ostream & os, lexp_nv_t const & v);

  // wrapper around a lexp_t that includes: (1) a map from key to
  // value and (2) a parent pointer. like a lexp_t itself, it may be a
  // leaf, if its src lexp_t is a leaf.
  struct lexp_name_val_map_t {
    p_lexp_t l; // source lexp. may be null (rarely, only for the no-value-init case). may be a leaf value.
    lexp_name_val_map_t * parent;
    bool nvm_init;
    std::map< sstr_t, p_lexp_t > nvm;
    void init_nvm( void );
    void dump( void );
    bool insert_leaf( char const * n, char const * v, bool const inc_use_cnt );
    void insert_leafs_from( vect_pair_str_str const & kvs );
    void insert_leafs_from( map_str_str const & kvs );
    p_lexp_t find( char const * n, lexp_name_val_map_t * * const found_scope ); // sets *found_scope 
    lexp_name_val_map_t( p_lexp_t l_ ) : l(l_), parent(0), nvm_init(0) { }
    lexp_name_val_map_t( p_lexp_t l_, lexp_name_val_map_t * parent_ ) : l(l_), parent(parent_), nvm_init(0) { }

  };
  std::ostream & operator <<(std::ostream & os, lexp_name_val_map_t const & v );

  typedef std::vector< lexp_nv_t > vect_lexp_nv_t;
  struct lexp_t
  {
    sstr_t src; // 'raw' source string. TODO: may be in cli format?
    bool src_has_trailing_comma( void ) const;
    sstr_t leaf_val; // if this exists, this is a leaf, and kids is empty and unused. 'cooked' value (escapes applied)
    vect_lexp_nv_t kids; // if !leaf_val.exists(), this is a list node. note: kids may be empty.
    uint32_t use_cnt; // scratch integer for client validation usage. after use, nodes with use_cn=0 generate errors.
    lexp_t( sstr_t const & src_ ) : src(src_), use_cnt(0) { }
    void get_stats( lexp_stats_t & lex_stats );
    p_lexp_t get_kid_by_name( char const * const n );
    void add_key_val( std::string const & k, std::string const & v );
    void add_key_lexp_val( std::string const & k, p_lexp_t const & v );
    void deep_inc_use_cnt( void );
#if 0 // unused   
    void add_all_kids_from( p_lexp_t lexp );
    void expand_include_xml( void ); 
#endif

  };
  std::ostream & operator <<(std::ostream & os, lexp_t const & v);

  p_lexp_t parse_lexp( std::string const & s );
  p_lexp_t parse_lexp_xml_file( std::string const & fn );
  p_lexp_t parse_lexp_list_xml( pugi::xml_node const & xn, std::string const & xml_fn );
  p_lexp_t parse_lexp_leaf_str( char const * const s );
  p_lexp_t make_list_lexp_from_one_key_val( std::string const & k, std::string const & v );
  p_lexp_t make_list_lexp_from_vals_vector( std::string const & k, vect_string const & vs );
  p_lexp_t get_lexp_from_argv( vect_string const & vs_argv, std::ostream & os );

  void test_lexp( void );
  void lexp_check_unused( lexp_t * l, vect_string & path );

  void str_format_from_nvm( string & out, string const & fmt, lexp_name_val_map_t & nvm );
  void str_format_from_nvm_str( string & out, string const & fmt, string const & nvm_str );

}

#endif /* _LEXP_H_ */
