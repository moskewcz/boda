#ifndef _LEXP_H_
#define _LEXP_H_

#include"boda_base.H"

namespace pugi { struct xml_node; }

namespace boda {

// sub-string pointer type with begin/end. references portion of larger shared string. uses p_uint8_t for base storage.
  struct sstr_t
  {
    p_uint8_t base;
    uint32_t b;
    uint32_t e;
    sstr_t( void ) : b(0), e(0) { }
    void set_from_string( std::string const &s );
    void borrow_from_string( std::string const &s );
    std::string str( void ) const { 
      assert(base); return std::string( base.get()+b, base.get()+e ); } // get copy as std::string
    uint32_t sz( void ) const { assert(base); assert( e >= b ); return e - b; }
    uint32_t end_off( void ) const { assert(base); return e; } 
    bool exists( void ) const { return base; }
    void shrink_end_off( uint32_t const new_e )
    {
      assert( new_e >= b ); // must yield sz >= 0
      assert( new_e <= e ); // must yield new_sz <= sz
      e = new_e;
    }
    bool operator < ( sstr_t const & o ) const;
  };
  std::ostream & operator<<(std::ostream & os, sstr_t const & v);
  struct lexp_t;
  struct lexp_stats_t; // for testing/internal use only
  typedef shared_ptr< lexp_t > p_lexp_t;
// name/value pair
  struct lexp_nv_t
  {
    sstr_t n;
    p_lexp_t v;
    void get_stats( lexp_stats_t & lex_stats );
  };
  std::ostream & operator<<(std::ostream & os, lexp_nv_t const & v);

  typedef std::map< sstr_t, p_lexp_t > lexp_name_val_map_t;

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
    void add_key_val( std::string const & k, std::string const & v );
  };
  std::ostream & operator<<(std::ostream & os, lexp_t const & v);

  p_lexp_t parse_lexp( std::string const & s );
  p_lexp_t parse_lexp_xml_file( std::string const & s );
  p_lexp_t parse_lexp_list_xml( pugi::xml_node const & xn );
  p_lexp_t make_list_lexp_from_one_key_val( std::string const & k, std::string const & v );

  void test_lexp( void );
  void lexp_check_unused( lexp_t * l, vect_string & path );

}

#endif /* _LEXP_H_ */
