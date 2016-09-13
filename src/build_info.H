#ifndef _BUILD_INFO_H_
#define _BUILD_INFO_H_
namespace boda
{
  char const * get_build_rev( void );
  char const * get_build_host( void );
  bool is_feature_enabled( char const * const feature_name );
}
#endif /* _BUILD_INFO_H_ */
