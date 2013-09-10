import os, re, socket;

def get_svn_rev():
    check_paths = [ '../pysrc', '../src/gen' ] # for proper operation, these paths must exists, so we check that
    out_py_fn = '../pysrc/svn_rev.py'
    out_c_fn = '../src/gen/svn_rev.c'

    for check_path in check_paths:
        if not os.path.isdir( check_path ):
            raise RuntimeError( repr(check_path) + " is not a dir, aborting to be safe ..." )

    info = os.popen( 'cd .. ; svn info' ).readlines()
    svn_rev = None
    for line in info:
        sline = re.split('[ ]+', line)
        if len(sline) == 2 and sline[0] == 'Revision:':
            svn_rev = int(sline[1])

    if svn_rev is None:
        raise RuntimeError( "couldn't parse svn info" )

    build_host = socket.gethostname()
    out_py = file( out_py_fn, 'w' )
    out_py.write( 'svn_rev = %s \n' % svn_rev )
    out_py.write( 'build_host = %r\n' % build_host )
    out_py.close();

    out_c = file( out_c_fn, 'w' )
    out_c.write( 'uint32_t const svn_rev = %s;\n' % svn_rev )
    out_c.write( 'char const * const build_host = %r;\n' % build_host )
    out_c.close();

    print "rev",svn_rev,"--> gen",[out_c_fn,out_py_fn],"<OK>"

