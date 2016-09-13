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

def get_build_info_c_str( enabled_features ): 
    # FIXME: if we want to support both svn and git, we should autodetect here?
    ret_template = """
#include<string>
#include<set> 
namespace boda {
  char const * get_build_rev( void ) { return "%(rev_str)s"; }
  char const * get_build_host( void ) { return "%(build_host)s"; }
  std::set< std::string > enabled_features = { %(enabled_features)s };
  bool is_feature_enabled( char const * const feature_name ) {
    return enabled_features.find( std::string( feature_name ) ) != enabled_features.end();
  }
}
"""
    return ret_template % { 
        "enabled_features" : ",".join( ['"%s"' % enabled_feature for enabled_feature in enabled_features ] ),
        "rev_str" : git_get_rev_str(),
        "build_host" : socket.gethostname(),
    }
    

