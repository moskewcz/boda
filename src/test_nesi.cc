#include"boda_tu_base.H"
#include"has_main.H"

namespace boda 
{
  typedef vector< double > vect_double;
  typedef shared_ptr< double > p_double;
  struct various_stuff_t : public virtual nesi, public has_main_t // NESI(help="test of various base types in nesi", bases=["has_main_t"], type_id="vst")
  {
    uint64_t u64; //NESI(help="unused u64",req=1)
    double dpf; //NESI(help="unused dbf",req=1)
    double dpf_nr; //NESI(help="unused dbf")
    vect_double vdpf; //NESI(help="unused dbf",req=1)
    p_double pdpf; //NESI(help="unused dbf",req=1)

    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( void ) {
      printf("tototo\n");
    }
  };

#include"gen/test_nesi.cc.nesi_gen.cc"
}

