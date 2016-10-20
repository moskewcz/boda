#ifndef _OP_BASE_H_
#define _OP_BASE_H_

#include"boda_base.H"

namespace boda 
{
  // op_base_t: information that defines the semantics of an operation
  struct op_base_t : virtual public nesi // NESI(help="op base descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    map_str_str str_vals; // NESI(default="()",help="per-operation-type specific parameters as a key/string-value list." )
    map_str_p_nda_t nda_vals; // NESI(default="()",help="per-operation-type specific parameters as a key/nda list." )
    op_base_t( void ) {}
    
    bool operator < ( op_base_t const & o ) const;
    // convenience access functions
    bool has( string const & an ) const;
    void set_dims( string const & an, dims_t const & dims ); // must not be set
    void set( string const & an, p_nda_t const & nda ); // must not be set
    void erase( string const & an ); // must already be set (not commonly used)
    void reset_dims( string const & an, dims_t const & dims ); // must already be set (not commonly used)
    dims_t const & get_dims( string const & an ) const;
    p_nda_t const & get( string const & an ) const;
    string const & get_str( string const & an ) const;
    uint32_t get_u32( string const & an ) const;
    void set_u32( string const & an, uint32_t const & v );
    bool operator ==( op_base_t const & o ) const = delete;
    bool operator !=( op_base_t const & o ) const { return (*this < o) || (o < *this); }

    // a few commonly-used str_vals get accessors for cleanliness/convenience. FIXME: should these be here?
    bool has_type( void ) const;
    string const & get_type( void ) const;
    void set_type( string const & type_ );
    void erase_type( void );

    bool has_func_name( void ) const;
    string const & get_func_name( void ) const;
    void set_func_name( string const & func_name_ );
    void erase_func_name( void );
  };
  typedef shared_ptr< op_base_t > p_op_base_t; 
  typedef set< op_base_t > set_op_base_t;

#include"nesi_decls.H"
  p_op_base_t make_p_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

}

#endif /* _OP_BASE_H_ */
