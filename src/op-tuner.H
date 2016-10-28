#ifndef _OP_TUNER_H_
#define _OP_TUNER_H_

#include"cnn_op.H"

namespace boda 
{


  // FIXME: should be NESI struct? see related FIXME on op_tune_wisdom_t
  struct op_run_t {
    string be_plat_tag;
    p_op_base_t op; // final function that was generated/run (note: *not* input op)
    double rt_secs;
    string err; // empty if no error
  };
  typedef vector< op_run_t > vect_op_run_t; 
  typedef map< string, op_run_t > map_str_op_run_t;

  // FIXME: yet-another place where we need/could-use generic NESI map str->nesi_struct support?
  struct op_tune_wisdom_t {
    p_op_tune_t op_tune;
    map_str_op_run_t runs; // map from be_plat_tag to op_run_t
    void merge_runs_from( op_tune_wisdom_t const & o );
  };
  typedef vector< op_tune_wisdom_t > vect_op_tune_wisdom_t; 
  typedef shared_ptr< op_tune_wisdom_t > p_op_tune_wisdom_t; 
  typedef vector< p_op_tune_wisdom_t > vect_p_op_tune_wisdom_t;

  // FIXME: since we don't have lt from op_tune_t, use lt on the str() of it ... not the best, but okay for now?
  struct by_op_tune_lt_p_op_tune_wisdom_t { bool operator()( p_op_tune_wisdom_t const & a, p_op_tune_wisdom_t const & b ) { 
    return str(a->op_tune) < str(b->op_tune); } };
  struct by_op_tune_set_p_op_tune_wisdom_t : public set< p_op_tune_wisdom_t, by_op_tune_lt_p_op_tune_wisdom_t > {
    void add_runs( vect_p_op_tune_wisdom_t const & wisdoms );
  };

  struct op_wisdom_t {
    p_op_base_t op;
    vect_pair_str_p_nda_digest_t kgs;
    vect_p_op_tune_wisdom_t wisdoms;
    void merge_wisdoms_from( op_wisdom_t const & o );
  };

  typedef vector< op_wisdom_t > vect_op_wisdom_t; 
  typedef shared_ptr< op_wisdom_t > p_op_wisdom_t; 
  typedef vector< p_op_wisdom_t > vect_p_op_wisdom_t;

  p_op_wisdom_t read_next_wisdom( p_istream const & in );
  void write_op_wisdom( op_wisdom_t const & op_wisdom, std::ostream & out );
  struct by_op_lt_p_op_wisdom_t { bool operator()( p_op_wisdom_t const & a, p_op_wisdom_t const & b ) { return *a->op < *b->op; } };
  typedef set< p_op_wisdom_t, by_op_lt_p_op_wisdom_t > by_op_set_p_op_wisdom_t;

}

#endif /* _OP_TUNER_H_ */
