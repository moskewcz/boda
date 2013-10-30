import os,sys,re
from get_svn_rev import get_svn_rev_c_str

join = os.path.join

def check_dir_writable( dir_name ):
    if not os.path.isdir( dir_name ) or not os.access( dir_name, os.W_OK ) :
        raise RuntimeError( "dir %r is not a writable directory." % (dir_name,) )

# basic info for pointer, vector, and basic/leaf types
class tinfo_t( object ):
    def __init__( self, tname, src_fn ):
        self.tname = tname
        self.src_fn = src_fn
        self.wrap_prefix = None
        self.wrap_type = None
        for prefix in ["p_","vect_"]:
            if tname.startswith( prefix ):
                self.wrap_prefix = prefix
                self.wrap_type = tname[len(prefix):]
    def get_tinfo( self ):
        gen_dict = { 'tname':self.tname, 'wrap_type':self.wrap_type }
        if self.wrap_prefix is None:
            return  ( 'tinfo_t tinfo_%(tname)s = { "%(tname)s", 0, nesi_%(tname)s_init, %(tname)s_make_p, ' +
                      '%(tname)s_vect_push_back };\n' ) % gen_dict
        elif self.wrap_prefix == 'p_':
            return ( 'tinfo_t tinfo_%(tname)s = { "%(tname)s", &tinfo_%(wrap_type)s, p_init, p_make_p, ' + 
                     'p_vect_push_back };\n' ) % gen_dict
        elif self.wrap_prefix == 'vect_':
            return ( 'tinfo_t tinfo_%(tname)s = { "%(tname)s", &tinfo_%(wrap_type)s, vect_init, vect_make_p, ' + 
                     'vect_vect_push_back };\n' ) % gen_dict
        else:
            raise RuntimeError( "bad wrap_prefix" + str(self.wrap_prefix) )

def iter_wrapped_types( tname ):
    for prefix in ["p_","vect_"]:
        if tname.startswith( prefix ):
            for wt in iter_wrapped_types( tname[len(prefix):] ):
                yield wt
    yield tname

class vinfo_t( object ):
    def __init__( self, tname, vname, help, req=0, default=None ):
        self.tname = tname
        self.vname = vname
        self.help = help
        self.req = req
        self.default = default
    def gen_vinfo( self ):
        default_val = "0"
        if self.default is not None:
            default_val = '"%s"' % (self.default)
        return '{ "%s", %s, %s, "%s", &tinfo_%s },\n' % ( self.help, default_val, self.req, self.vname, self.tname )

class cinfo_t( object ):
    def __init__( self, cname, src_fn, help, bases=[], type_id=None, is_abstract=None, tid_vn=None ):
        self.cname = cname
        self.src_fn = src_fn
        self.help = help
        self.bases = bases
        self.type_id = type_id
        self.is_abstract = is_abstract
        self.tid_vn = tid_vn
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
        ret = 'vinfo_t vinfos_%s[] = {' % self.cname
        for vi in self.vars_list:
            ret += vi.gen_vinfo()
        ret += '{} };\n'
        return ret
    def gen_cinfos_list( self, base_or_derived ):
        cis = getattr(self,base_or_derived)
        ret =  "".join( "extern cinfo_t cinfo_" + ci.cname + '; ' for ci in cis )
        ret += 'cinfo_t * cinfos_%s_%s[] = {' % (base_or_derived, self.cname)
        ret += "".join( "&cinfo_%s," % (ci.cname) for ci in cis )
        ret += '0 };\n'
        return ret
    def gen_cinfo( self ):
        tid_vix = "uint32_t_const_max"
        if not self.tid_vn is None:
            for (ix,vi) in enumerate(self.vars.itervalues()):
                if vi.vname == self.tid_vn:
                    tid_vix = ix
                    break
            else:
                raise RuntimeError("in nesi struct %r: %r specified as tid_nv, but no nesi var with that name found." %
                                   (self.cname,self.tid_vn) )
        tid_str = "0"
        if not self.type_id is None:
            tid_str = '"'+self.type_id+'"'
            
        concrete_new_template = """
  void * make_p_%(cname)s( void * v ) { 
    shared_ptr< %(cname)s > *pv = (shared_ptr< %(cname)s > *)v;
    pv->reset( new %(cname)s ); 
    return pv->get(); 
  }
  void * vect_push_back_%(cname)s( void * v ) { 
    vector< %(cname)s > * vv = ( vector< %(cname)s > * )( v );
    vv->push_back( %(cname)s() ); return &vv->back(); 
  }
"""
        abstract_new_template = """
  void * make_p_%(cname)s( void * v ) { rt_err("can't create abstract class %(cname)s"); }
  void * vect_push_back_%(cname)s( void * v ) { rt_err("can't create abstract class %(cname)s"); }
"""
        new_template = concrete_new_template
        if self.is_abstract: new_template = abstract_new_template
        gen_template = ""
        gen_template += self.gen_cinfos_list( "bases" )
        gen_template += self.gen_cinfos_list( "derived" )
        gen_template += new_template + """
  cinfo_t cinfo_%(cname)s = { "%(cname)s", "%(help)s", nesi__%(cname)s__get_field, 
    vinfos_%(cname)s, make_p_%(cname)s, %(tid_vix)s, %(tid_str)s, cinfos_derived_%(cname)s, cinfos_bases_%(cname)s };
  tinfo_t tinfo_%(cname)s = { "%(cname)s", &cinfo_%(cname)s, nesi_struct_init, nesi_struct_make_p, vect_push_back_%(cname)s };

"""
        return gen_template % {'cname':self.cname, 'help':self.help, 'num_vars':len(self.vars),
                               'tid_vix':tid_vix, 'tid_str':tid_str }


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
        self.tinfos_seen = set()
        self.tinfos = []

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
            
        # convert cinfo.bases from names to objects and fill in bases and derived types lists
        for cinfo in self.cinfos.itervalues():
            base_names = cinfo.bases
            cinfo.bases = []
            for btn in base_names:
                bt = self.cinfos.get(btn,None)
                if bt is None:
                    raise RuntimeError( "NESI type %r listed %r as its base_type, but that is not a NESI type" %
                                        (cinfo.cname,btn) )
                cinfo.bases.append( bt )
                bt.derived.append( cinfo )

        # populate tinfos and cinfos for NESI structs
        #for cinfo in self.cinfos.itervalues():
        #    self.tinfos.setdefault( cinfo.cname, tinfo_t( cinfo.src_fn, len(self.tinfos), cinfo.cname ) )
            
        # populate tinfos for any remaining types (vector, pointer, and base types)
        for cinfo in self.cinfos.itervalues():
            for var in cinfo.vars_list:
                wts = list( iter_wrapped_types( var.tname ) )
                assert len(wts)
                lt = wts[0]
                src_fn = None
                if lt in self.cinfos:
                    src_fn = self.cinfos[lt].src_fn
                    wts[0:1] = []
                else:
                    src_fn = "nesi.cc"
                for wt in wts:
                    if wt in self.tinfos_seen:
                        continue
                    self.tinfos_seen.add( wt )
                    self.tinfos.append( tinfo_t( wt, src_fn ) )

        # create per-file generated code files
        per_file_gen = {}
        for cinfo in self.cinfos.itervalues():
            gf = per_file_gen.setdefault( cinfo.src_fn, ['#include "../nesi_decls.H"\n'] )
            gf.append( cinfo.gen_get_field() )
            gf.append( cinfo.gen_vinfos_predecls() )
            gf.append( cinfo.gen_vinfos() )
            gf.append( cinfo.gen_cinfo() )

        for tinfo in self.tinfos:
            gf = per_file_gen.setdefault( tinfo.src_fn, ['#include "../nesi_decls.H"\n'] )
            gf.append( tinfo.get_tinfo() )
        
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
    def proc_sdecl( self, src_fn, cname, **kwargs ):
        if cname in self.cinfos:
            raise RuntimeError( "duplicate NESI struct declaration for %r" % cname )
        self.cur_sdecl = cinfo_t( cname, src_fn, **kwargs )
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
