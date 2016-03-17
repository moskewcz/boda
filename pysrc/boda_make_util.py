# Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
import os, re

# one complication/flaw of the gcc generated .d files + the make
# system we are using is that when a source file is changed in a way
# that *removes* a dependency on a header file *and* that header file
# is removed, make will refuse to rebuild the .cc file since the
# exisiting .d file references a header that no longer exists. to deal
# with this, we add rules with no dependencies or recipies for each
# unique header file. this causes make to ignore the dependency on
# such headers if they do not exist. as a (good) side effect, if a
# header that is really needed does not exist, we'll always defer to
# the compiler to generate an error rather than maybe having make give
# an error if the header is mentioned in the .d file.
class DepFileProc( object ):
    def __init__( self ):
        self.all_header_fns = set()
        for root, dirs, fns in os.walk( "." ):
            for fn in fns:
                if fn.endswith(".d"): self.proc_d_fn( fn )
            dirs[:] = []
        out = open('make_make_ignore_missing_headers_in_d_files.make','w')
        for header_fn in self.all_header_fns:
            out.write( header_fn + ":\n" )
        out.close()
        #print self.all_header_fns
    def proc_d_fn( self, fn ):
        lines = open(fn).readlines()
        if not lines: raise ValueError( "empty .d file: " + fn )
        # we expect each .d file is single-logical-line make rule split with "\\\n" pairs into lines
        rule = ""
        for l in lines[:-1]:
            assert( l.endswith("\\\n") )
            rule += l[:-2]
        rule += lines[-1] # may end with newline, we don't care
        parts = rule.split()
        assert len(parts) >= 2 # we assume there is at least a target and one dep
        assert parts[0].endswith('.o:') # maybe too strong, but should end with ':' for sure (or we're parsing wrong)
        # parts[1] should be the source file, not checked (or checkable?)
        for header_fn in parts[1:]: self.all_header_fns.add( header_fn )


ospj = os.path.join

class DepInfo( object ): 
    def __init__( self ): 
        self.use_cnt = 0
        self.enable = 0
        self.force_disable = 0
        self.lines = []
        self.needs = []
        self.gen_fns = []
    def make_enabled( self, dep_name, deps ):
        if self.enable: return # already enabled
        if self.force_disable:
            raise ValueError( "can't enable force-disabled dependency %s (via sub-dep)." % (dep_name,) );
        self.enable = 1
        for sub_dep in self.needs: deps[sub_dep].make_enabled( sub_dep, deps ) # recurse

class GenObjList( object ):

    def next_line( self ):
        while 1:
            assert self.next_ol_line <= len(self.ol_lines)
            if self.next_ol_line == len(self.ol_lines):
                self.cur_line = None
                return False
            line = self.ol_lines[self.next_ol_line]
            self.next_ol_line += 1
            comment_char_idx = line.find('#')
            if comment_char_idx != -1: line = line[:comment_char_idx] # remove comment if any
            orig_line = line
            line = line.strip()
            if not line: continue # skip blank lines
            self.cur_line = line
            self.cur_orig_line = orig_line # unstripped
            return True

    def parse_error( self, msg ):
        assert self.cur_line is not None
        raise ValueError( "obj_list parse error on line %s: %s\nline was:%s" %( self.next_ol_line, msg, self.cur_line) );

    def get_section_start( self ):
        if self.cur_line is None: return None
        sec_re_match = re.match( '\[\s*(.+)\s*\]$', self.cur_line )
        if not sec_re_match: return None
        return sec_re_match.group(1).split()

    def at_section_start_or_eof( self ):
        if self.cur_line is None: return True
        return not ( self.get_section_start() is None )

    def parse_objs( self ):
        while not self.at_section_start_or_eof():
            parts = self.cur_line.split()
            assert parts
            enable = True
            for dep_name in parts[1:]:
                neg_dep = dep_name[0] == '-'
                if neg_dep: dep_name = dep_name[1:] # strip neg if present
                if not dep_name in self.deps: 
                    self.parse_error( "unknonwn dep name %r for object (note: dep section must preceed usage)" % (dep_name,) )
                dep = self.deps[dep_name]
                if dep.force_disable ^ neg_dep: enable = False
                elif not neg_dep: dep.use_cnt += 1
            if enable: self.gen_objs.append( parts[0] )
            self.next_line();
    
    def parse_dep( self, sec_start ):
        dep_name = sec_start[0]
        if dep_name in self.deps: self.parse_error( "duplicate dep section" )
        new_dep = DepInfo()
        needs_str = "needs="
        gen_fn_str = "gen_fn="
        for opt in sec_start[1:]:
            if 0: pass
            elif opt == "disable": new_dep.force_disable = 1
            elif opt.startswith( needs_str ): new_dep.needs.append( opt[len(needs_str):] )
            elif opt.startswith( gen_fn_str ): new_dep.gen_fns.append( opt[len(gen_fn_str):] )
            else: self.parse_error( "unknown dep section option %r" % (opt,) )
        while not self.at_section_start_or_eof():
            new_dep.lines.append( self.cur_orig_line ) # note: whitespace preserved
            self.next_line();
        self.deps[dep_name] = new_dep

    def __init__( self ):    
        self.deps = {}
        self.gen_fns = set()
        self.gen_objs = []
        self.proj_root_dir = '..'
        ol_fn = ospj(self.proj_root_dir,"obj","obj_list")
        self.ol_lines = open( ol_fn ).readlines()
        if not self.ol_lines: raise ValueError( "empty obj_list file at: " + ol_fn )
        self.next_ol_line = 0
        self.next_line()

        while self.cur_line is not None:
            sec_start = self.get_section_start()
            if sec_start is None: self.parse_error( "expecting start of section" )
            self.next_line() # consume section start line
            if sec_start[0] == "objs": self.parse_objs()
            else: self.parse_dep( sec_start )
        
        if not self.gen_objs: raise ValueError( "obj_list error: [objs] section missing or empty" )
            
        # add in any generated c++ file that need top-level compilation. FIXME: probably this shouldn't be hard-coded here.
        self.gen_objs.append( 'build_info.o' ) 
        gen_objs = open('gen_objs','w')
        gen_objs.write( ''.join( gen_obj + '\n' for gen_obj in self.gen_objs ) )
        gen_objs.close()

        # enable directly used deps (and recursivly thier needs/sub-dep usages)
        for dep_name, dep in self.deps.iteritems(): 
            if dep.use_cnt != 0:
                dep.make_enabled( dep_name, self.deps ) 

        dep_make = open('dependencies.make','w')
        for dep_name, dep in self.deps.iteritems():
            dep_make.write( "# generated make lines for dependency: %s use_cnt=%s force_disable=%s\n" % 
                            (dep_name,dep.use_cnt,dep.force_disable) )
            if not dep.enable: 
                dep_make.write( "# not enabled, skipping.\n" )
                continue 
            for line in dep.lines: dep_make.write( line )
            for gen_fn in dep.gen_fns: self.gen_fns.add( gen_fn )
        dep_make.close();
