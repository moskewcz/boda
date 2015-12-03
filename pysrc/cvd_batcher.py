import shlex, subprocess, time, argparse, os

class Slot( object ):
    def __init__( self, cudid ):
        self.cudid = cudid
        self.proc = None

    def is_done( self ):
        assert self.proc
        if self.proc.poll() is not None:
            self.finish_job()
            self.proc = None
            return True
        return False

    def finish_job( self ):
        assert self.proc
        assert self.proc.returncode is not None
        self.proc.wait()
        job_out_lines = open( self.job_out_fn ).readlines()
        for line in job_out_lines: print self.job_out_fn + ": " + line,

    def start_job( self, job, job_id ):
        assert self.proc is None
        self.job_out_fn = "boda_job_%s.txt" % job_id
        job_out = open( self.job_out_fn, "w" )
        job_out.write( job + "\n" )
        print "starting="+self.job_out_fn+" "+job
        job_env = dict(os.environ)
        job_env["CUDA_VISIBLE_DEVICES"] = str(self.cudid)
        self.proc = subprocess.Popen(shlex.split(job),stdout=job_out,env=job_env)
        assert self.proc

def poll_running( running_slots, open_slots ):
    assert len(running_slots)
    cur_running_slots = list(running_slots)
    running_slots[:] = []
    for slot in cur_running_slots: 
        if slot.is_done():
            open_slots.append(slot)
        else:
            running_slots.append(slot)
    time.sleep(0.1)
        


def nin_maxs():
    pass
    cccp1_max_out_sz_1=2745.04
    cccp2_max_out_sz_1=2417.22
    cccp3_max_out_sz_1=3390.36
    cccp4_max_out_sz_1=2379.46
    cccp5_max_out_sz_1=1417.23
    cccp6_max_out_sz_1=1036.65
    cccp7_max_out_sz_1=429.797
    cccp8_max_out_sz_1=265.939
    conv1_max_out_sz_1=2175.99
    conv2_max_out_sz_1=4496.18 # note, a bit above 4096 ...
    conv3_max_out_sz_1=3587.47
    conv4_max_out_sz_1=1072.84 # a bit over 1024
    data_max_out_sz_1=151
    pool0_max_out_sz_1=2417.22
    pool2_max_out_sz_1=2379.46
    pool3_max_out_sz_1=1036.65
    pool4_max_out_sz_1=53.0894

gn_maxs = """cls1_fc1_max_out_sz_1=35.8276
cls1_fc2_max_out_sz_1=50.2411
cls1_pool_max_out_sz_1=575.587
cls1_reduction_max_out_sz_1=228.299
cls2_fc1_max_out_sz_1=36.0346
cls2_fc2_max_out_sz_1=57.8928
cls2_pool_max_out_sz_1=551.419
cls2_reduction_max_out_sz_1=194.549
cls3_fc_max_out_sz_1=58.746
cls3_pool_max_out_sz_1=42.8646
conv1_max_out_sz_1=2570.95
conv2_max_out_sz_1=914.679
data_max_out_sz_1=151
icp1_out0_max_out_sz_1=548.291
icp1_out1_max_out_sz_1=1070.58
icp1_out2_max_out_sz_1=1008.41
icp1_out3_max_out_sz_1=412.056
icp1_pool_max_out_sz_1=138.726
icp1_reduction1_max_out_sz_1=537.009
icp1_reduction2_max_out_sz_1=508.725
icp2_in_max_out_sz_1=1070.58
icp2_out0_max_out_sz_1=730.086
icp2_out1_max_out_sz_1=1071.36
icp2_out2_max_out_sz_1=1097
icp2_out3_max_out_sz_1=780.042
icp2_out_max_out_sz_1=1097
icp2_pool_max_out_sz_1=1070.58
icp2_reduction1_max_out_sz_1=859.086
icp2_reduction2_max_out_sz_1=814.854
icp3_in_max_out_sz_1=1097
icp3_out0_max_out_sz_1=1007.7
icp3_out1_max_out_sz_1=2005.87
icp3_out2_max_out_sz_1=911.565
icp3_out3_max_out_sz_1=1119.64
icp3_out_max_out_sz_1=2005.87
icp3_pool_max_out_sz_1=1097
icp3_reduction1_max_out_sz_1=1484.64
icp3_reduction2_max_out_sz_1=964.933
icp4_out0_max_out_sz_1=989.744
icp4_out1_max_out_sz_1=1017.98
icp4_out2_max_out_sz_1=1033.57
icp4_out3_max_out_sz_1=1274.94
icp4_out_max_out_sz_1=1274.94
icp4_pool_max_out_sz_1=2005.87
icp4_reduction1_max_out_sz_1=1324.8
icp4_reduction2_max_out_sz_1=1051.6
icp5_out0_max_out_sz_1=934.511
icp5_out1_max_out_sz_1=776.28
icp5_out2_max_out_sz_1=1033.07
icp5_out3_max_out_sz_1=813.397
icp5_out_max_out_sz_1=1033.07
icp5_pool_max_out_sz_1=1274.94
icp5_reduction1_max_out_sz_1=716.224
icp5_reduction2_max_out_sz_1=1167.12
icp6_out0_max_out_sz_1=612.964
icp6_out1_max_out_sz_1=825.474
icp6_out2_max_out_sz_1=711.703
icp6_out3_max_out_sz_1=897.455
icp6_out_max_out_sz_1=897.455
icp6_pool_max_out_sz_1=1033.07
icp6_reduction1_max_out_sz_1=894.621
icp6_reduction2_max_out_sz_1=670.333
icp7_out0_max_out_sz_1=573.903
icp7_out1_max_out_sz_1=443.347
icp7_out2_max_out_sz_1=378.679
icp7_out3_max_out_sz_1=546.675
icp7_out_max_out_sz_1=573.903
icp7_pool_max_out_sz_1=897.455
icp7_reduction1_max_out_sz_1=503.249
icp7_reduction2_max_out_sz_1=575.445
icp8_in_max_out_sz_1=573.903
icp8_out0_max_out_sz_1=367.577
icp8_out1_max_out_sz_1=304.327
icp8_out2_max_out_sz_1=293.447
icp8_out3_max_out_sz_1=437.182
icp8_out_max_out_sz_1=437.182
icp8_pool_max_out_sz_1=573.903
icp8_reduction1_max_out_sz_1=373.101
icp8_reduction2_max_out_sz_1=451.395
icp9_out0_max_out_sz_1=131.121
icp9_out1_max_out_sz_1=106.018
icp9_out2_max_out_sz_1=108.681
icp9_out3_max_out_sz_1=124.847
icp9_out_max_out_sz_1=131.121
icp9_pool_max_out_sz_1=437.182
icp9_reduction1_max_out_sz_1=215.652
icp9_reduction2_max_out_sz_1=245.483
norm1_max_out_sz_1=138.726
norm2_max_out_sz_1=138.726
pool1_max_out_sz_1=2570.95
pool2_max_out_sz_1=138.726
reduction2_max_out_sz_1=388.774
"""

import math

def add_googlenet_conv_jobs( jobs ):
    linfo = []
    li_lines = gn_maxs.split()    
    for li_line in li_lines:
        (ln,v) = li_line.split("=")
        if ln.startswith("data") or ln.startswith("cls"): continue
        v = int(math.ceil(float(v)))
        #kb = 1
        #while (2**kb) < v: kb += 1
        suf = "_max_out_sz_1"
        assert ln.endswith(suf)
        ln = ln[:-len(suf)]
        #print ln,kb
        linfo.append( (ln,v) )
    qopts_base = 'keep_bits=%(bits)s,quantize=('
    for li in linfo: qopts_base += "_=(name=%s,max_val=%s)," % li
    qopts_base += ')'
    for i in range(16):
        job = 'boda test_lmdb --model-name=googlenet_conv --num-to-read=50000 --run-cnet="(in_dims=(img=20),ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=cls3_fc-conv,compute_mode=1,conv_fwd=(mode=nvrtc,enable_stats=0,show_rtc_calls=0,'+(qopts_base%{'bits':i})+'))"'
        jobs.append( job )

def add_nin_imagenet_jobs( jobs ):
    linfo = [ ("conv1",4096),("cccp1",4096), ("cccp2",4096), 
              ("conv2",4096),("cccp3",4096), ("cccp4",4096),
              ("conv3",4096),("cccp5",2048), ("cccp6",2048), 
              ("conv4",1024),("cccp7",512), ("cccp8",512), ]
    qopts_base = 'keep_bits=%(bits)s,quantize=('
    for li in linfo: qopts_base += "_=(name=%s,max_val=%s)," % li
    qopts_base += ')'
    for i in range(16):
        job = 'boda test_lmdb --model-name=nin_imagenet --num-to-read=50000 --run-cnet="(in_dims=(img=20),ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=pool4,compute_mode=1,conv_fwd=(mode=nvrtc,enable_stats=0,show_rtc_calls=0,'+(qopts_base%{'bits':i})+'))"'
        jobs.append( job )

def add_alexnet_ng_conv_jobs( jobs ):
    qopts_base = 'keep_bits=%(bits)s,quantize=(_=(name=conv1,max_val=4096),_=(name=conv2,max_val=1024),_=(name=conv3,max_val=1024),_=(name=conv4,max_val=512),_=(name=conv5,max_val=512))'
    for i in range(16):
        job = 'boda test_lmdb --model-name=alexnet_ng_conv --num-to-read=50000 --run-cnet="(in_dims=(img=20),ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=fc8-conv,compute_mode=1,conv_fwd=(mode=nvrtc,enable_stats=0,show_rtc_calls=0,'+(qopts_base%{'bits':i})+'))"'
        jobs.append( job )

def main():
    parser = argparse.ArgumentParser(description='spawn multiple boda processes.')
    parser.add_argument('--num-devices', metavar='N', type=int, default=8, help='number of devices to use in queue')
    args = parser.parse_args()

    jobs = []
    
    add_googlenet_conv_jobs( jobs )
    # add_nin_imagenet_jobs( jobs )
    # add_alexnet_ng_conv_jobs( jobs )

    open_slots = []
    running_slots = []
    for i in range(args.num_devices):
        ns = Slot( i )
        open_slots.append( ns )

    job_id = 0
    for job in jobs:
        while not len(open_slots): poll_running( running_slots, open_slots )
        slot = open_slots.pop()
        slot.start_job( job, job_id )
        job_id += 1
        running_slots.append( slot )
    
    while len(running_slots): poll_running( running_slots, open_slots )
        

if __name__ == "__main__":
    main()
