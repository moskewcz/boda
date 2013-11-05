#include"boda_tu_base.H"
#include"has_main.H"

namespace boda 
{
  struct various_stuff_t;
  typedef shared_ptr< various_stuff_t > p_various_stuff_t;
  typedef vector< p_various_stuff_t > vect_p_various_stuff_t;
  typedef vector< double > vect_double;
  typedef vector< uint64_t > vect_uint64_t;
  typedef shared_ptr< double > p_double;
  typedef shared_ptr< string > p_string;
  struct one_p_string_t : public virtual nesi // NESI(help="struct with one p_string")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_string s; // NESI(help="foo")
  };
  typedef vector< one_p_string_t > vect_one_p_string_t;
  struct various_stuff_t : public virtual nesi, public has_main_t // NESI(help="test of various base types in nesi", bases=["has_main_t"], type_id="vst")
  {
    uint64_t u64; //NESI(help="unused u64",req=1)
    double dpf; //NESI(help="unused dbf",req=1)
    double dpf_nr; //NESI(help="unused dbf")
    vect_double vdpf; //NESI(help="unused dbf",req=1)
    p_double pdpf; //NESI(help="unused dbf",req=1)
    vect_uint64_t vu64; //NESI(help="unused dbf",req=1)
    vect_p_various_stuff_t vvs; //NESI(help="unused dbf")
    vect_one_p_string_t vops; //NESI(help="unused")

    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( void ) {
      printf("tototo\n");
    }
  };

#include"gen/test_nesi.cc.nesi_gen.cc"
}

