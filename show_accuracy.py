from __future__ import division, print_function
import os
import argparse
parser = argparse.ArgumentParser(description="show accuracy")
parser.add_argument("--method", type=str, default="drop_connect")
parser.add_argument("--log-dir", type=str, default="logs")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.log_dir, "%s.log" % args.method)) as f:
        accuracies = map(lambda x: float(x[-9:-1]), f.readlines())
    print(max(accuracies))