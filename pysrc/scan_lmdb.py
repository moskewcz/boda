
# Roughly approximates some of Symas microbenchmark.

import multiprocessing
import os
import random
import shutil
import sys
import tempfile
import time

try:
    import affinity
except:
    affinity = False
import lmdb


USE_SPARSE_FILES = sys.platform != 'darwin'
DB_PATH = sys.argv[1]
MAX_KEYS = int(4e6)

def open_env():
    return lmdb.open(DB_PATH,
        map_size=1048576 * 1024,
        metasync=False,
        sync=False,
        map_async=True,
        writemap=USE_SPARSE_FILES)


env = open_env()
#with env.begin() as txn:
#    keys = list(txn.cursor().iternext(values=False))

with env.begin() as txn:
    cursor = txn.cursor()
    for key,value in cursor:
        vl = len(value)
        if 1 or vl != 196624:
            print key, len(value)

env.close()
