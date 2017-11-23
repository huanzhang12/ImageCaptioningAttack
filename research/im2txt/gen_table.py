#!/usr/bin/env python3

import os
import sys
import glob
import argparse

def check_lines(lines, pattern_lists):
    res = {}
    for l in lines:
        for pattern in pattern_lists:
            pattern_l = pattern.lower().replace('-', '_')
            if l.lower().startswith(pattern_l):
                number = float(l.split(":")[1])
                res[pattern] = number
    return res

def process_file_list(filelist):
    results = {}
    for fn in filelist:
        names = fn.split('_')
        c = int(names[1])
        kappa = int(names[2])
        with open(fn) as f:
            lines = f.readlines()
            res = check_lines(lines, patterns)
            # skip empty files
            if res:
                if c not in results:
                    results[c] = {}
                results[c][kappa] = res
                print("Processed {}: c={}, kappa={}, results={}".format(fn, c, kappa, res))
            else:
                print("Skipped {}".format(fn))
    return results
            
patterns = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE_L", "METEOR"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="directory of log file for processing")
    args = parser.parse_args()

    adv_ori_csv = glob.glob(os.path.join(args.log_dir, 'transfer_*_results_adv_ori.log'))
    adv_tgt_csv = glob.glob(os.path.join(args.log_dir, 'transfer_*_results_adv_tgt.log'))

    adv_ori_results = process_file_list(adv_ori_csv)
    adv_tgt_results = process_file_list(adv_tgt_csv)

    for pattern in patterns:
        sys.stdout.write("\multicolumn{1}{l|}{\\!\\!" + pattern + "}" + " " * (17 - len(pattern)))
        for kappa in [1, 5, 10]:
            for c in [10, 100, 1000]:
                sys.stdout.write("&  {}     &  {}        ".format("{:.3f}".format(adv_ori_results[c][kappa][pattern]).lstrip('0'), "{:.3f}".format(adv_tgt_results[c][kappa][pattern]).lstrip('0')))
        print("&")

