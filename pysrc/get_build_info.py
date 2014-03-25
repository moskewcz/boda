# Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
import os, re, socket;

# unused/untested (used to work, but was refactored during transition to git)
def svn_get_rev_str():
    info = os.popen( 'cd .. ; svn info' ).readlines()
    svn_rev = None
    for line in info:
        sline = re.split('[ ]+', line)
        if len(sline) == 2 and sline[0] == 'Revision:':
            svn_rev = int(sline[1])
    if svn_rev is None:
        raise RuntimeError( "couldn't parse svn info" )
    return svn_rev

def git_get_rev_str():
    info = os.popen( 'cd .. ; git rev-parse --verify HEAD --short=6' ).readlines()
    if len(info) != 1:
        raise RuntimeError( "couldn't parse git rev info, expected a single line as output, got:" + info )
    git_rev = info[0].strip()
    if not git_rev:
        raise RuntimeError( "couldn't parse git rev info, returned line was only whitespace:" + repr(info) )
    return git_rev

def get_build_info_c_str(): 
    # FIXME: if we want to support both svn and git, we should autodetect here?
    rev_str = git_get_rev_str();
    build_host = socket.gethostname()
    ret = ""
    ret += 'char const * const build_rev = "%s";\n' % rev_str 
    ret += 'char const * const build_host = "%s";\n' % build_host
    return ret

