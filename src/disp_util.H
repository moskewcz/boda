#ifndef _DISP_UTIL_H_
#define _DISP_UTIL_H_

namespace boda {
  struct img_t;
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t;
  void disp_skel( vect_p_img_t const & imgs );
}

#endif /* _DISP_UTIL_H_ */
