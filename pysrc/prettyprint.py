# pretty printing stuff. factor out somewhere?
def pad( v, s ): return s + " "*max(0,v - len(s))
def pp_val_part( v, force ):
    if v < 10: return "%.2f" % v
    if v < 100: return "%.1f" % v
    if (v < 1000) or force: return "%.0f" % v
    return None
def pp_val( v ): # pretty-print flops
    exp = 0
    assert v >= 0
    if v == 0: return "0"
    while v < 1.0:
        v *= 1000.0
        exp -= 1
    ret = pp_val_part( v, 0 )
    while ret is None:
        v /= 1000.0
        exp += 1
        ret = pp_val_part( v, exp == 5 )
    if exp < -4: return str(v) # too small, give up
    #print "v",v,"exp",exp
    if exp < 0: return ret+"munp"[- 1 - exp]
    if exp == 0: return ret
    return ret+"KMGTP"[exp - 1]

def pp_pc( v ): return "%.1f%%" % (v*100.0,)
verbose_print = 0
if verbose_print:
    def pp_secs( v ): return pp_val( v ) + " SECS"
    def pp_flops( v ): return pp_val( v ) + " FLOPS"
    def pp_bytes( v ): return pp_val( v ) + " BYTES"
    def pp_bps( v ): return pp_val( v ) + " BYTES/SEC"
    def pp_fpb( v ): return pp_val( v ) + " FLOPS/BYTE"
    def pp_fps( v ): return pp_val( v ) + " FLOPS/SEC"
    def pp_fpspw( v ): return pp_val( v ) + " FLOPS/SEC/WATT"
    def pp_joules( v ): return pp_val( v ) + " JOULES"
else:
    def pp_secs( v ): return pp_val( v ) + "s"
    def pp_flops( v ): return pp_val( v ) + "F"
    def pp_bytes( v ): return pp_val( v ) + "B"
    def pp_bps( v ): return pp_val( v ) + "B/s"
    def pp_fpb( v ): return pp_val( v ) + "F/B"
    def pp_fps( v ): return pp_val( v ) + "F/s"
    def pp_fpspw( v ): return pp_val( v ) + "F/s/W"
    def pp_joules( v ): return pp_val( v ) + "J"

