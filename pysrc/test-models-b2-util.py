import sys,os

# example full b2 'friendly' URL for model download:
# https://f001.backblazeb2.com/file/boda-models-v1/models/alexnet_ng_conv/best.caffemodel

# run from root of boda working copy; with no options, will download models:
# moskewcz@maaya:~/git_work/boda$ python pysrc/test-models-b2-util.py 

def main( args ):
    if not os.access( args.net_names_fn, os.R_OK ):
        raise ValueError( "can't read net names file %r; this utility should be run from boda's root directory." % 
                          (args.net_names_fn,) )
    for net_name in open(args.net_names_fn).readlines():
        b2_net_fn = "models/"+net_name.strip()+"/best.caffemodel"
        if args.upload:
            local_net_fn = "/scratch/" + b2_net_fn
            cmd = "b2 upload_file %s %s %s" % (args.bucket, local_net_fn, b2_net_fn)
        else:
            b2_friendly_url = args.dl_url + b2_net_fn
            # assume that we want to put the file in exactly b2_net_fn relative to the current directory
            cmd = "curl %s --create-dirs -o %s" % (b2_friendly_url,b2_net_fn)
        print "running:", cmd
        if not args.dry_run: os.system( cmd )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='generate list of operations.')
    parser.add_argument('--bucket', metavar="BUCKET", type=str, default="boda-models-v1", help="name of b2 bucket (used for upload only)" )
    parser.add_argument('--net-names-fn', metavar="FN", type=str, default="test/test_all_nets.txt", help="filename for list of net names file (upload or download)" )
    parser.add_argument('--upload', metavar="BOOL", type=bool, default=0, help="if 1, upload instead of download" )
    parser.add_argument('--dry-run', metavar="BOOL", type=bool, default=0, help="if 1, just print commands, don't run them" )
    parser.add_argument('--dl-url', metavar="STR", type=str, default="https://f001.backblazeb2.com/file/boda-models-v1/", help="download URL prefix (used for download only; note: includes bucket name)" )

    args = parser.parse_args()

    main(args)
