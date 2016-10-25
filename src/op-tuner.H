#ifndef _OP_TUNER_H_
#define _OP_TUNER_H_

#include"cnn_op.H"

namespace boda 
{


  // FIXME: should be NESI struct? see related FIXME on op_tune_wisdom_t
  struct op_run_t {
    string be_plat_tag;
    p_op_base_t op;
    double rt_secs;
    string err; // empty if no error
  };
  typedef vector< op_run_t > vect_op_run_t; 
  typedef map< string, op_run_t > map_str_op_run_t;

  // FIXME: yet-another place where we need/could-use generic NESI map str->nesi_struct support?
  struct op_tune_wisdom_t {
    p_op_tune_t op_tune;
    map_str_op_run_t runs; // map from be_plat_tag to op_run_t
  };
  typedef vector< op_tune_wisdom_t > vect_op_tune_wisdom_t; 
  typedef shared_ptr< op_tune_wisdom_t > p_op_tune_wisdom_t; 
  typedef vector< p_op_tune_wisdom_t > vect_p_op_tune_wisdom_t;

  struct op_wisdom_t {
    p_op_base_t op;
    vect_pair_str_p_nda_digest_t kgs;
    vect_p_op_tune_wisdom_t wisdoms;
  };

  typedef vector< op_wisdom_t > vect_op_wisdom_t; 
  typedef shared_ptr< op_wisdom_t > p_op_wisdom_t; 
  typedef vector< p_op_wisdom_t > vect_p_op_wisdom_t;

  p_op_wisdom_t read_next_wisdom( p_istream const & in );
  void write_op_wisdom( op_wisdom_t const & op_wisdom, std::ostream & out );

}

#endif /* _OP_TUNER_H_ */
