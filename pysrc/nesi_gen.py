import os,sys,re
from get_svn_rev import get_svn_rev_c_str

join = os.path.join

def check_dir_writable( dir_name ):
    if not os.path.isdir( dir_name ) or not os.access( dir_name, os.W_OK ) :
        raise RuntimeError( "dir %r is not a writable directory." % (dir_name,) )

# basic info for all types
class tinfo_t( object ):
    def __init__( self, src_fn, tid, tname ):
        self.src_fn = src_fn
        self.tname = tname
        self.tid = tid
    def get_tinfo( self ):
        return 'tinfo_t tinfo_%s = { "%s" };\n' % (self.tname, self.tname)

def iter_wrapped_types( tname ):
    for prefix in ["p_","vect_"]:
        if tname.startswith( prefix ):
            iter_wrapped_types( tname[len(prefix):] )
    yield tname

class cinfo_t( object ):
    def __init__( self, cname, src_fn, help, base_type=None, type_id=None, is_abstract=None ):
        self.cname = cname
        self.src_fn = src_fn
        self.help = help
        self.base_type = base_type
        self.type_id = type_id
        self.is_abstract = is_abstract
        self.vars = {}
        self.vars_list = []
        self.derived = [] # cinfos's of directly derived types


    def gen_get_field( self ):
        get_field_template =  """
  // gen_field for %(cname)s
  void * nesi__%(cname)s__get_field( void * const o, uint32_t const ix )
  {
    %(skip_to)s%(cname)s * const to = (%(cname)s *)( o );
    switch( ix ) {
%(field_cases)s   
    }
    //rt_err( "NESI internal error: invalid field ix=" + str(ix) ); // fancy
    assert( !"NESI internal error: invalid field ix" ); // less-so
  }
"""
        skip_to = ""
        if not self.vars: skip_to = "// "
        field_cases = ""
        for ix,var in enumerate( self.vars_list ):
            field_cases += "      case %s: return &to->%s;\n" % (ix,var.vname)
        return get_field_template % { "skip_to":skip_to, "cname":self.cname, "field_cases":field_cases }

    def gen_vinfos_predecls( self ):
        return ( ''.join( [ ( 'extern tinfo_t tinfo_%s; ' % ( vi.tname ) ) 
                            for vi in self.vars_list ]) + '\n')

    def gen_vinfos( self ):
        return ( ('vinfo_t vinfos_%s[] = {' % self.cname) + ''.join( 
                [ ( '{ "%s", "%s", &tinfo_%s },\n' % ( vi.help, vi.vname, vi.tname ) ) 
                  for vi in self.vars_list ]) + '};\n' )

    def gen_cinfo( self ):
        concrete_new_template = """
  void * newt_%(cname)s( void ) { return new %(cname)s; }
  void * vect_push_back_%(cname)s( void * v ) { 
    vector< %(cname)s > * vv = ( vector< %(cname)s > * )( v );
    vv->push_back( %(cname)s() ); return &vv->back(); 
  }
"""
        abstract_new_template = """
  void * newt_%(cname)s( void ) { rt_err("can't create abstract class %(cname)s"); }
  void * vect_push_back_%(cname)s( void * v ) { rt_err("can't create abstract class %(cname)s"); }
"""
        new_template = concrete_new_template
        if self.is_abstract: new_template = abstract_new_template
        gen_template = new_template + """
  cinfo_t cinfo_%(cname)s = { "%(help)s", nesi__%(cname)s__get_field, %(num_vars)s, vinfos_%(cname)s};
  tinfo_t tinfo_%(cname)s = { "%(cname)s", &cinfo_%(cname)s, nesi_struct_init, newt_%(cname)s, vect_push_back_%(cname)s };

"""
        return gen_template % {'cname':self.cname, 'help':self.help, 'num_vars':len(self.vars) }

class vinfo_t( object ):
    def __init__( self, tname, vname, help, req=0, default=None ):
        self.tname = tname
        self.vname = vname
        self.help = help
        self.req = req
        self.default = default


class nesi_gen( object ):

    def __init__( self ):
        # note: gen dir should contain only generated files, so that stale
        # generated files can be detected/removed. that is, any file
        # present but not generated inside gen_dir will be removed.
        self.src_dir = join('..','src')
        self.gen_dir = join(self.src_dir,'gen')
        self.gen_objs = []
        self.gen_fns = set()

        # codegen data
        self.cinfos = {}
        self.tinfos = {}

        try:
            os.mkdir( self.gen_dir )
        except OSError, e:
            pass
        check_dir_writable( self.gen_dir )

        # scan all files and cache NESI commands
        for root, dirs, fns in os.walk( self.src_dir ):
            for fn in fns:
                self.proc_fn( fn )
            dirs[:] = []
            
        # fill in derived types lists
        for cinfo in self.cinfos.itervalues():
            if cinfo.base_type is not None:
                bt = self.cinfos.get(cinfo.base_type,None)
                if bt is None:
                    raise RuntimeError( "NESI type %r listed %r as its base_type, but that is not a NESI type" %
                                        (cinfo.cname,cinfo.base_type) )
                cinfo.derived.append( bt )

        # populate tinfos and cinfos for NESI structs
        #for cinfo in self.cinfos.itervalues():
        #    self.tinfos.setdefault( cinfo.cname, tinfo_t( cinfo.src_fn, len(self.tinfos), cinfo.cname ) )
            
        # populate tinfos for any remaining types (vector, pointer, and base types)
        for cinfo in self.cinfos.itervalues():
            for var in cinfo.vars_list:
                self.tinfos.setdefault( var.tname, tinfo_t( None, len(self.tinfos), var.tname ) )

        # create tinfos
        tinfos_text = []
        for tinfo in self.tinfos.itervalues():
            tinfos_text.append( tinfo.get_tinfo() )

        self.update_file_if_different( 'tinfos.cc', "".join(tinfos_text) )
                
        # create per-file generated code files
        per_file_gen = {}
        for cinfo in self.cinfos.itervalues():
            gf = per_file_gen.setdefault( cinfo.src_fn, ['#include "../nesi_decls.H"'] )
            gf.append( cinfo.gen_get_field() )
            gf.append( cinfo.gen_vinfos_predecls() )
            gf.append( cinfo.gen_vinfos() )
            gf.append( cinfo.gen_cinfo() )
        
        for gfn,gfn_texts in per_file_gen.iteritems():
            self.update_file_if_different( gfn+'.nesi_gen.cc', "".join( gfn_texts ) )

        
            
        self.update_file_if_different( 'svn_rev.cc', get_svn_rev_c_str() )
        self.gen_objs.append( 'svn_rev.o' )
        self.update_file_if_different( 'gen_objs', ''.join( gen_obj + '\n' for gen_obj in self.gen_objs ) )

        self.remove_stale_files()
        print "wrappers up to date."

    def print_err_eval( self, s ):
        try:
            return eval(s)
        except Exception, e:
            raise RuntimeError("NESI eval failed for %r: %s" % (s, e) )

    def proc_fn( self, fn ):
        if not ( fn.endswith(".cc") or fn.endswith(".H") ):
            return # skip file if it's not a boda source or header file. note that this skips the lodepng/pugixml src.
        fn_path = join( self.src_dir, fn )
        fn_lines = file( fn_path ).readlines()
        # FIXME: add multiline support
        # for now, any lines matching this re are considered NESI command lines. this is certainly not perfect/complete.
        nesi_decl = re.compile( r"//\s*NESI\s*\((.*)\)" ) # note: greedy, which is what we want
        struct_decl = re.compile( r"\s*(?:class|struct)\s*(\S+)" ) # NESI structs decl lines must match this
        var_decl = re.compile( r"\s*(\S+)\s*(\S+)\s*;" )  # NESI var decls lines must match this
        self.cur_sdecl = None
        for line in fn_lines:
            nd_ret = nesi_decl.search(line)
            if nd_ret:
                nesi_kwargs = nd_ret.groups()[0]
                sd_ret = struct_decl.match(line)
                if sd_ret:
                    cname = sd_ret.groups()[0]
                    # note: sets self.cur_sdecl
                    self.print_err_eval( "self.proc_sdecl(%r, %r,%s)" % (fn,cname,nesi_kwargs) ) 
                    continue
                vd_ret = var_decl.match(line)
                if vd_ret:
                    tname, vname = vd_ret.groups()
                    self.print_err_eval( "self.proc_vdecl(%r,%r,%s)" % (tname,vname,nesi_kwargs) ) 
                    continue
                raise RuntimeError( "line looks like NESI decl, but doesn't match as struct or var decl:" + line )

    # for now, we support only a (very) limited form of
    # inheritance/factory generation: we generate a single factory
    # function returning p_base_type for each unique base type
    # mentioned across all sdecls. the factory will instantiate the
    # class with a type_id the matches the input's top-level type
    # field.
    def proc_sdecl( self, src_fn, cname, help, base_type=None, type_id=None, is_abstract=None ):
        if cname in self.cinfos:
            raise RuntimeError( "duplicate NESI struct declaration for %r" % cname )
        self.cur_sdecl = cinfo_t( cname, src_fn, help, base_type, type_id, is_abstract )
        self.cinfos[cname] = self.cur_sdecl
        
    def proc_vdecl( self, tname, vname, help, req=0, default=None ):
        if self.cur_sdecl is None:
            raise RuntimeError( "NESI var declaration for var %r %r before any NESI struct declaration" % (tname,vname) )
        if vname in self.cur_sdecl.vars:
            raise RuntimeError( "duplicate NESI var declaration for %r in struct %r " % (vname,self.cur_sdecl.cname) )
        vi = vinfo_t( tname, vname, help, req, default )
        self.cur_sdecl.vars[vname] = vi
        self.cur_sdecl.vars_list.append( vi )
            
    def update_file_if_different( self, gen_fn, new_file_str ):
        assert( not os.path.split( gen_fn )[0] ) # gen_fn should have no path components (i.e. just a filename)
        if gen_fn in self.gen_fns:
            raise RuntimeError( "tried to generate file %s more than once" % gen_fn )
        self.gen_fns.add( gen_fn )
        targ = join( self.gen_dir, gen_fn )
        try:
            cur_file_str = file(targ).read()
            if cur_file_str == new_file_str:
                return
            print "out_of_date:",targ
        except IOError, e:
            print "read_failed:",targ
        print "regenerating:",targ
        file(targ,"w").write(new_file_str)

    def remove_stale_files( self ):
        to_remove = []
        for root, dirs, fns in os.walk( self.gen_dir ):
            for fn in fns:
                if fn not in self.gen_fns:
                    to_remove.append( fn )
            dirs[:] = []
        if to_remove:
            print "removing state generated files:", " ".join( to_remove ) 
            for fn in to_remove:
                os.unlink( join( self.gen_dir, fn ) )
