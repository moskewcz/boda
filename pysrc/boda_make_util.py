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
            line = line.strip()
            if not line: continue # skip blank lines
            self.cur_line = line
            return True

    def parse_error( self, msg ):
        assert self.cur_line is not None
        raise ValueError( "obj_list parse error on line %s: %s\nline was:%s" %( self.next_ol_line, msg, self.cur_line) );

    def get_section_start( self ):
        if self.cur_line is None: return None
        sec_re_match = re.match( '\[\s*(.+)\s*\]$', self.cur_line )
        if not sec_re_match: return None
        return sec_re_match.group(1)

    def at_section_start_or_eof( self ):
        if self.cur_line is None: return True
        return not ( self.get_section_start() is None )

    def parse_objs( self ):
        while not self.at_section_start_or_eof():
            self.gen_objs.append( self.cur_line )
            self.next_line();
    
    def parse_dep( self, dep_name ):
        if dep_name in self.deps: self.parse_error( "duplicate dep section" )
        dep_lines = []
        while not self.at_section_start_or_eof():
            dep_lines.append( self.cur_line )
            self.next_line();
        self.deps[dep_name] = dep_lines

    def __init__( self ):    
        self.deps = {}
        self.gen_objs = []
        self.src_dir = ospj('..','src')
        ol_fn = ospj(self.src_dir,"obj_list")
        self.ol_lines = open( ol_fn ).readlines()
        if not self.ol_lines: raise ValueError( "empty obj_list file at: " + ol_fn )
        self.next_ol_line = 0
        self.next_line()

        while self.cur_line is not None:
            sec_name = self.get_section_start()
            if sec_name is None: self.parse_error( "expecting start of section" )
            self.next_line() # consume section start line
            if sec_name == "objs": self.parse_objs()
            else: self.parse_dep( sec_name )
        
        if not self.gen_objs: raise ValueError( "obj_list error: [objs] section missing or empty" )
            
        # add in any generated c++ file that need top-level compilation. FIXME: probably this shouldn't be hard-coded here.
        self.gen_objs.append( 'build_info.o' ) 
        gen_objs = open('gen_objs','w')
        gen_objs.write( ''.join( gen_obj + '\n' for gen_obj in self.gen_objs ) )
        gen_objs.close()

        dep_make = open('dependencies.make','w')
        for dep_name,dep_lines in self.deps.iteritems():
            dep_make.write( "# generated make lines for dependency: " + dep_name + "\n" )
            for line in dep_lines: dep_make.write( line + "\n" )
        dep_make.close();
