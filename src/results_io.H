#include"boda_base.H"

namespace boda
{
  using namespace std;

  // currently internal use only, but maybe can/should be exported
  //void read_results_file( p_img_db_t img_db, std::string const & fn, std::string const &class_name );
  //p_img_db_t read_image_list_file( std::string const & fn );

  void score_results_file( std::string const & il_fn, std::string const & res_fn, std::string const &class_name );
}
