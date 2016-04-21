CUCL_DEVICE float det_hash_rand( uint32_t const rv ) {
  uint32_t h = rv;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return (float)( h ) * ( 10.0f / (float)( U32_MAX ) ) - 5.0f;
}
