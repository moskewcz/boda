import sys

# note: uses pystrace from https://github.com/dirtyharrycallahan/pystrace

# example usage: first, get strace log from running all tests:
# moskewcz@maaya:~/git_work/boda/run/tr1$ strace -f -ttt -T -o st.out boda test_all
# then, process the log to get a list of files opened (for now, we don't check if they were opened successfully or not):
# moskewcz@maaya:~/git_work/boda/run/tr1$ cat st.out | python ../../pysrc/strace-ana.py | sort -u > test-all-files.txt

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
