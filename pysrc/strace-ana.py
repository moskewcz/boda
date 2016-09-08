import sys

# note: uses pystrace from https://github.com/dirtyharrycallahan/pystrace

from pystrace.strace import *

def main( args ):
    strace_stream = StraceInputStream(sys.stdin)
    for entry in strace_stream:            
        if entry.syscall_name == "open":
            fn = entry.syscall_arguments[0]
            assert( fn[0] == '"' and fn[-1] == '"' )
            print( fn[1:-1] )

if __name__ == "__main__":
	main(sys.argv[1:])
