#!/usr/bin/env python
import sys
import lmdb

env =  lmdb.open( sys.argv[1], map_size=1048576 * 1024, metasync=False, sync=False, map_async=True, writemap=True )

with env.begin() as txn:
    cursor = txn.cursor()
    for key,value in cursor:
        vl = len(value)
        if 1 or vl != 196624:
            print key, len(value)

env.close()
