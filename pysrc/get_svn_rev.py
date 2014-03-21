# Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
import os, re, socket;

def get_svn_rev_c_str():
    info = os.popen( 'cd .. ; svn info' ).readlines()
    svn_rev = None
    for line in info:
        sline = re.split('[ ]+', line)
        if len(sline) == 2 and sline[0] == 'Revision:':
            svn_rev = int(sline[1])

    if svn_rev is None:
        raise RuntimeError( "couldn't parse svn info" )

    build_host = socket.gethostname()
    ret = ""
    ret += 'unsigned const svn_rev = %s;\n' % svn_rev 
    ret += 'char const * const build_host = "%s";\n' % build_host
    return ret
