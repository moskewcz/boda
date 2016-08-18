#ifndef _OP_BASE_H_
#define _OP_BASE_H_

#include"boda_base.H"

namespace boda 
{
  // op_base_t: information that defines the semantics of an operation
  struct op_base_t : virtual public nesi // NESI(help="op base descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string type; //NESI(help="op type; determines overall semantics and locates named operation metadata",req=1)
    map_str_str str_vals; // NESI(default="()",help="per-operation-type specific parameters as a key/string-value list." )
    map_str_p_nda_t nda_vals; // NESI(default="()",help="per-operation-type specific parameters as a key/nda list." )
    op_base_t( void ) {}
    // used only for stats/quantize ... remove?
    op_base_t( string const & type_, map_str_dims_t const & dims_vals_, map_str_str const & str_vals_ );
    
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
    bool operator ==( op_base_t const & o ) const = delete;
    bool operator !=( op_base_t const & o ) const = delete;
  };
  typedef shared_ptr< op_base_t > p_op_base_t; 
  typedef set< op_base_t > set_op_base_t;
}

#endif /* _OP_BASE_H_ */
