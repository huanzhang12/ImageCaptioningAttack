from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import argparse
import glob
import sys
sys.path.append('./show-attend-and-tell')
from run_inference import CaptionInference
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_size", default=3, help="number of candidates from log")
    parser.add_argument("--quiet", action="store_true", help="don't print sentence to stdout")
    parser.add_argument("--output", default="transfer_sentences", help="save output to a file")
    parser.add_argument("log_dir", type=str, help="directory of log file for processing")
    args = parser.parse_args()

    # looking for all record_*.csv in log dir
    csv_files = glob.glob(os.path.join(args.log_dir, 'record_*.csv'))

    print(csv_files)
    # read all csv files into a pandas dataframe
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    result = pd.concat(dfs)

    total_attacks = result.shape[0]
    # filter out unsuccessful images
    result = result[result['attack successful?'] == True]
    success_attacks = result.shape[0]
    print("success {}/{} = {}".format(success_attacks, total_attacks, success_attacks / total_attacks))

    # get relavent fields
    file_names = list(result['attack filename'])
    adv_file_names = [os.path.join(args.log_dir, 'adversarial_' + n.replace("jpg", "png") + ".npy") for n in file_names]
    ori_file_names = [os.path.join(args.log_dir, 'original_' + n.replace("jpg", "png") + ".npy") for n in file_names]
    attack_sentences = []
    for i in range(args.beam_size):
        attack_sentences.append(list(result['target caption {}'.format(i + 1)]))
    l2_dist = list(result['L2 distortion'])
    linf_dist = list(result['L_inf distortion'])

    print("Average L2:", sum(l2_dist) / len(l2_dist))
    print("Average Linf:", sum(linf_dist) / len(linf_dist))

    # inference on the show-attend-and-tell model
    with tf.Session() as sess:
        # create the caption generator
        cap_infer = CaptionInference(sess, 'show-attend-and-tell/model_best/model-best', True)
        ori_captions = cap_infer.inference_files(ori_file_names)
        adv_captions = cap_infer.inference_files(adv_file_names)

    # prepare output files
    f_ori = open(args.output + "_ori.txt", "w")
    f_adv = open(args.output + "_adv.txt", "w")
    f_tgt = open(args.output + "_tgt.txt", "w")

    # print to output
    for i in range(len(ori_captions)):
        image_id = os.path.splitext(file_names[i])[0].split('_')[-1]
        f_ori.write("{}\t{}\n".format(image_id, ori_captions[i]))
        f_adv.write("{}\t{}\n".format(image_id, adv_captions[i]))
        if not args.quiet:
            print('original    :', ori_captions[i])
            print('adversarial :', adv_captions[i])
        for j in range(args.beam_size):
            s = attack_sentences[j][i]
            f_tgt.write("{}\t{}\n".format(image_id, s))
            if not args.quiet:
                print('adv target {}: {}'.format(j, s))
        if not args.quiet:
            print('---------------------')

    f_ori.close()
    f_adv.close()
    f_tgt.close()

