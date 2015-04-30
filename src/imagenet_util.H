#ifndef _IMAGENET_UTIL_H_
#define _IMAGENET_UTIL_H_

namespace boda {
  uint8_t const inmc_r = 123U;
  uint8_t const inmc_g = 117U;
  uint8_t const inmc_b = 104U;
  uint32_t const u32_rgba_inmc = uint32_t(inmc_r)+(uint32_t(inmc_g)<<8)+(uint32_t(inmc_b)<<16)+(255U<<24); // RGBA
  uint32_t const u32_bgra_inmc = uint32_t(inmc_b)+(uint32_t(inmc_g)<<8)+(uint32_t(inmc_r)<<16)+(255U<<24); // BGRA

}

#endif /* _IMAGENET_UTIL_H_ */
