#!/usr/bin/env python3
import sys
import os
import subprocess
from collections import defaultdict

sysde = sys.getdefaultencoding()

class GetDeps(object):
    def __init__(self, args):
        self.args = args
        self.libnames = set()
        self.needs = defaultdict( set )
        self.get_link_libs()
        self.get_ldd_deps()
        self.get_manual_pkgs()

    def get_link_libs(self):
        ldflags = open(args.LDFLAGS_fn).read()
        ldflags = ldflags.split()
        for ldflag in ldflags:
            if not ldflag.startswith('-l'):
                continue
            libname = "lib"+ldflag[2:]
            self.libnames.add(libname)
        
    def get_ldd_deps(self):
        ldd_out = subprocess.run( ["ldd",self.args.exe_fn], check=True, stdout=subprocess.PIPE ).stdout
        ldd_out = ldd_out.decode(sysde).split("\n")
        for ldd_line in ldd_out:
            parts = ldd_line.split()
            if (not len(parts)) or (parts[0] == "linux-vdso.so.1"): continue
            for i,p in enumerate(parts):
                if p == "=>": self.proc_dep( parts[i+1] )

    def proc_dep(self, dep):
        if dep[0] != os.path.sep:
            raise ValueError( "dep %r doesn't start with path seperator (i.e. isn't an absolute path), refusing to handle" % dep )
        if not os.path.isfile(dep):
            raise ValueError( "dep %r isn't a regular file" % dep )
        # optionally, check if we explictly linked to this file
        
        self.file_get_pgk(dep)
        
    def file_get_pgk(self, fn):
        try:
            out = subprocess.run( ["dpkg","-S",fn], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE ).stdout
            out = out.decode(sysde).split(":")
            self.needs[out[0]].add(fn)
        except subprocess.CalledProcessError as e:
            err_out = e.stderr.decode(sysde)
            if err_out.startswith( "dpkg-query: no path found matching pattern" ):
                self.needs["no-package-found"].add(fn)
            else:
                raise

    def get_manual_pkgs(self):
        #print(self.needs)
        npf = self.needs.pop( "no-package-found", None )
        if npf:
            print("files-shown-as-deps-by-ldd-not-in-any-package:")
            print( " ".join( npf ))
        print("packages-containing-files-shown-as-deps-by-ldd:")
        print( " ".join( self.needs.keys() ) )
        
import argparse
parser = argparse.ArgumentParser(description='use ldd and apt to get list of debian/ubuntu packages needed to run an exe')
parser.add_argument('--exe-fn', default="../../lib/boda", metavar="FN", type=str, help="exe to analyze")
parser.add_argument('--LDFLAGS-fn', default="../../obj/LDFLAGS.txt", metavar="FN", type=str, help="list of link-time libs (i.e. LDFLAGS)")

args = parser.parse_args()
gd = GetDeps(args)

# example output from 2017-12-08, run on boda:
# libv4l-0 libsndfile1 libgflags2v5 libuuid1 libwavpack1 libswscale-ffmpeg3 libharfbuzz0b libatlas3-base libdatrie1 libvorbis0a libxcursor1 liblz4-1 libsz2 libkrb5support0 libboost-thread1.58.0 libgmp10 libv4lconvert0 libwayland-cursor0 libpng12-0 libtinfo5 libexpat1 libgpg-error0 libconsole-bridge0.2v5 libswresample-ffmpeg1 libjasper1 libshine3 libpython2.7 libboost-system1.58.0 libpangoft2-1.0-0 libtasn1-6 libavutil-ffmpeg54 libx11-6 libjson-c2 libroslz4-0d libleveldb1v5 libx264-148 libc6 libsoxr0 libbsd0 libilmbase12 libopenjpeg5 libpulse0 libwayland-client0 libice6 libk5crypto3 libgfortran3 libtwolame0 libgomp1 libjbig0 libaec0 libsndio6.1 libpixman-1-0 libavformat-ffmpeg56 libpango-1.0-0 libfreetype6 libhdf5-10 libxcomposite1 libtiff5 libgme0 libvorbisenc2 libxrandr2 libsnappy1v5 libboost-regex1.58.0 nvidia-384 libxcb-render0 libgcrypt20 libbluray1 libopencv-highgui2.4v5 libxext6 libp11-kit0 libzvbi0 libboost-iostreams1.58.0 libxau6 libmodplug1 libdc1394-22 libtbb2 libcairo2 libsm6 libgsm1 libcpp-common0d libssh-gcrypt-4 libogg0 libsdl2-ttf-2.0-0 libwrap0 libunwind8 libglib2.0-0 zlib1g libopencv-core2.4v5 libroscpp-serialization0d libasyncns0 libvpx3 libpangox-1.0-0 libfontconfig1 libxvidcore4 libicu55 libquadmath0 libschroedinger-1.0-0 librostime0d libflac8 libgssapi-krb5-2 libxdmcp6 libxrender1 libtheora0 libffi6 libcuda1-384 libglu1-mesa librtmp1 libgtk2.0-0 libatk1.0-0 libgtkglext1 libva1 libkrb5-3 libjpeg-turbo8 libbz2-1.0 libboost-filesystem1.58.0 liblmdb0 liborc-0.4-0 libudev1 libopenexr22 libxdamage1 libxt6 libidn11 libxml2 libwebp5 libx265-79 libgraphite2-3 libdbus-1-3 libxi6 libraw1394-11 libxss1 libpangocairo-1.0-0 liborocos-kdl1.3 libboost-program-options1.58.0 libgcc1 libgnutls30 libxcb1 libcomerr2 libnettle6 libgdk-pixbuf2.0-0 libnuma1 liblzma5 libedit2 libthai0 libxmu6 libhogweed4 libllvm3.8 libstdc++6 libxkbcommon0 libsdl2-2.0-0 libavcodec-ffmpeg56 libwayland-egl1-mesa libselinux1 libspeex1 libcrystalhd3 libmp3lame0 libasound2 libkeyutils1 libxxf86vm1 libpcre3 nvidia-driveworks libsystemd0 libtf2-0d libxfixes3 libopus0 libgoogle-glog0v5 libusb-1.0-0 librosbag-storage0d libxcb-shm0 libopencv-imgproc2.4v5

# example output from 2017-12-08, run on obj-disp/boda:
#files-shown-as-deps-by-ldd-not-in-any-package:
#/mnt/nfs/exp/lib64/libGLEW.so.2.1 /mnt/nfs/exp/lib/libOSMesa.so.8 /usr/lib/nx/X11/Xinerama/libXinerama.so.1 /mnt/nfs/exp/lib/libglapi.so.0 /mnt/nfs/exp/lib/libprotobuf.so.13
#packages-containing-files-shown-as-deps-by-ldd:
#libgcc1 libxml2 libfontconfig1 libboost-system1.58.0 liborc-0.4-0 libsdl2-ttf-2.0-0 libasound2 libdbus-1-3 libpcre3 libxkbcommon0 libedit2 libk5crypto3 libselinux1 libgssapi-krb5-2 libflac8 zlib1g libgsm1 libxext6 libjson-c2 librtmp1 libvorbis0a libxau6 libgpg-error0 libopenjpeg5 libshine3 libcrystalhd3 libavcodec-ffmpeg56 libvpx3 libkrb5-3 libcomerr2 libmodplug1 libnuma1 libxfixes3 libhogweed4 libgme0 libxss1 libasyncns0 libkrb5support0 libicu55 libboost-iostreams1.58.0 libbsd0 libsndio6.1 libwrap0 libwayland-cursor0 libllvm3.8 libexpat1 libsnappy1v5 libboost-filesystem1.58.0 libtinfo5 liblzma5 libmp3lame0 libx265-79 libxxf86vm1 libidn11 libavformat-ffmpeg56 libgomp1 libwayland-egl1-mesa libxi6 libva1 libgnutls30 libboost-program-options1.58.0 libboost-regex1.58.0 libbluray1 libxcursor1 libzvbi0 libx11-6 libc6 libavutil-ffmpeg54 libopus0 libffi6 libgcrypt20 libxrandr2 libssh-gcrypt-4 libp11-kit0 libgmp10 libsoxr0 libtasn1-6 libsdl2-2.0-0 libsystemd0 libstdc++6 libpulse0 libfreetype6 libwavpack1 libbz2-1.0 libpng12-0 libwebp5 libtwolame0 libkeyutils1 libwayland-client0 libnettle6 libsndfile1 libswresample-ffmpeg1 libpython2.7 libxvidcore4 libxdmcp6 libxcb1 libvorbisenc2 libogg0 libx264-148 libschroedinger-1.0-0 libxrender1 libspeex1 libtheora0

