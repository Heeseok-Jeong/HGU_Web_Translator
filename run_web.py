#-*-coding: utf-8-*-
from flask import Flask, render_template, request, url_for
import subprocess
import sys
import argparse
import os
import torch
import torch.nn as nn
from nmt_main import train_model, translate_file
from deeplib.text_data import TextPairIterator, TextIterator, read_dict
from nmt_model import translate_beam_k, translate_beam_1
from deeplib.utils import timeSince, ids2words, unbpe
from subprocess import Popen, PIPE, check_output, call
from io import open
from bible_people.convert_PN import convert_pn_for_web
import pickle
import operator

# 하이퍼파라미터 세팅
parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--max_length", type=int, default=100) #Max length of input src
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--src_file", type=str, default='')
parser.add_argument("--kr_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--en_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--k2e_model_file", type=str, default='')
parser.add_argument("--e2k_model_file", type=str, default='')
parser.add_argument("--beam_width", type=int, default=1)

EOS_token = 1

args = parser.parse_args()
k2e_model = ""
e2k_model = ""
k2e_trg_inv_dict = ""
e2k_trg_inv_dict = ""
PN_dict = ""
PN_list = ""
PN_dict_name = 'bible_people/bible_peo+loc.pkl'
lang = "k2e"


# app 객체로 라우팅 경로 설정
app = Flask(__name__)

# 모델 세팅 
def setting():
    print("Setting models")
    torch.no_grad()

    args.kr_dict='bible_people/bible_data_GYGJ+NIV/PN/subword/vocab.kr.pkl'
    args.en_dict='bible_people/bible_data_GYGJ+NIV/PN/subword/vocab.en.pkl'
    args.save_dir = './results'
    args.k2e_model_file = 'kr2en.mylstm.150.250.250.250.bible_data_GJNIV_PN'
    args.e2k_model_file = 'en2kr.mylstm.300.500.500.500.bleu05'
    args.src_file = '/home/nmt19/RNN_model/input.txt.tok.sym.pn.sub'

    kr_dict = read_dict(args.kr_dict)
    en_dict = read_dict(args.en_dict)
    with open(PN_dict_name, 'rb') as f:
        PN_dict = pickle.load(f, encoding="utf-8")

    str_temp =""
    count = 0
    for kk, vv in PN_dict.items():
        if(len(kk) == 1):
            count += 1
            str_temp += kk
            str_temp += "\n"
    one_char_name = "bible_people/name_one_char.txt"
    with open(one_char_name, 'w') as f:
        f.write(str_temp)

    key_file = open(one_char_name, "r", encoding="utf-8")

    key_line = key_file.readline()
    while key_line:
        key_line = key_line.replace('\n', '')
        PN_dict.pop(key_line)
        key_line = key_file.readline()

    key_file.close()

    PN_list= sorted(PN_dict.items(), key=operator.itemgetter(1), reverse=True)

    k2e_trg_inv_dict = dict()
    for kk, vv in en_dict.items():
        k2e_trg_inv_dict[vv] = kk

    e2k_trg_inv_dict = dict()
    for kk, vv in kr_dict.items():
        e2k_trg_inv_dict[vv] = kk

    k2e_model_name = args.save_dir + '/' + args.k2e_model_file + '.pth' + '.best.pth'
    e2k_model_name = args.save_dir + '/' + args.e2k_model_file + '.pth' + '.best.pth'

    k2e_model = torch.load(k2e_model_name)
    print("k2e best model loaded")
    
    e2k_model = torch.load(e2k_model_name)
    print("e2k best model loaded")
    
    return k2e_model, e2k_model, k2e_trg_inv_dict, e2k_trg_inv_dict, PN_list

# 번역 페이지 실행
@app.route("/")
def hello():
    return render_template("k2e.html")

# 한->영 번역 
@app.route('/k2e_trans', methods=['POST', 'GET'])
def k2e_trans(num=None):
    if request.method == 'GET':
        return render_template("k2e.html")
    if request.method == 'POST':
        if request.form['src'] == "":
            return render_template("k2e.html")
        input_sen = request.form['src']
        replaced_sen = ""

        # 토큰화
        text_file = open("input.txt", "w", encoding="utf8")
        text_file.write(input_sen)
        text_file.close()
        tokenizer=check_output('./tokenizer.perl en < input.txt> input.txt.tok',shell=True)

        # 숫자 기호화 '1 => __N0'
        number_sym=call('./web_symbolize.py',shell=True)
        text_file = open("input.txt.tok.sym", "r", encoding="utf8")
        replaced_sen = text_file.read()
        print("number_sym : ", replaced_sen)

        # 고유명사 기호화 'Trumph => __P0'
        lang = "k2e"
        replaced_sen, info_dict = convert_pn_for_web(replaced_sen, PN_list, lang)
        print("replaced_sen : " + replaced_sen)
        print("info_dict : ", info_dict)
        text_file.close()


        text_file = open("input.txt.tok.sym.pn", "w", encoding="utf8")
        text_file.write(replaced_sen)
        text_file.close()

        apply_bpe=check_output("../subword_nmt/apply_bpe.py -c " +\
                                "./bible_people/bible_data_GYGJ+NIV/PN/subword/kr.5000.code " +\
                                "< ./input.txt.tok.sym.pn > ./input.txt.tok.sym.pn.sub", shell=True)

        # k2e 모델에 입력
        valid_iter = TextIterator(args.src_file, args.kr_dict,
                                 batch_size=1, maxlen=1000,
                                 ahead=1, resume_num=0)
        for x_data, x_mask, cur_line, iloop in valid_iter:
            samples = translate_beam_1(k2e_model, x_data, args)
            output = ids2words(k2e_trg_inv_dict, samples, eos_id=EOS_token)
            output = unbpe(output)

        output = output.replace(" &apos; ", "\'")
        output = output.replace(" &apos;", "\'")
        output = output.replace("&apos; ", "\'")
        output = output.replace("&apos;", "\'")
        output = output.replace(" &quot; ", "\"")
        output = output.replace(" &quot;", "\"")
        output = output.replace("&quot; ", "\"")
        output = output.replace("&quot;", "\"")

        # 숫자 기호화 되돌리기
        mapping=open("mapping.sym","rb")
        num_dict=pickle.load(mapping)

        print("num_dict : ", num_dict)
        print("output1 : " + output)
        for key, value in num_dict.items(): #key : __NO / value : 25
            if key in output:
                output=output.replace(key,value)

        # 고유명사 기호화 되돌리기
        for key, val in info_dict.items(): #key : __P0, val : 예수(한국어)
            temp = key.strip()
            if temp in output:
                for (PN_key, PN_val) in PN_list :
                    if val == PN_key:
                        output = output.replace(temp, PN_val)

        return render_template('k2e.html', src_contents = input_sen, trans_contents = output)
    
    else:
        return render_template("k2e.html")

# 영->한 번역 수행 
@app.route('/e2k_trans', methods=['POST', 'GET'])
def e2k_trans(num=None):
    if request.method == 'GET':
        return render_template("e2k.html")
    if request.method == 'POST':
        if request.form['src'] == "":
            return render_template("e2k.html")
        input_sen = request.form['src']

        text_file = open("input.txt", "w", encoding="utf8")
        text_file.write(input_sen)
        text_file.close()

        tokenizer=check_output('./tokenizer.perl en < input.txt> input.txt.tok',shell=True)
        apply_bpe=check_output('../subword_nmt/apply_bpe.py -c ../data_05/bleu05/test/kr.10000.code < ./input.txt.tok > ./input.txt.tok.sym.sub',shell=True)

        valid_iter = TextIterator(args.src_file, args.en_dict,
                                 batch_size=1, maxlen=1000,
                                 ahead=1, resume_num=0)
        for x_data, x_mask, cur_line, iloop in valid_iter:
            samples = translate_beam_1(e2k_model, x_data, args)
            output = ids2words(e2k_trg_inv_dict, samples, eos_id=EOS_token)
            output = unbpe(output)

        output = output.replace(" &apos; ", "\'")
        return render_template('e2k.html', src_contents = input_sen, trans_contents = output)
    else:
        return render_template("e2k.html")


# 서버 구동 
host_addr = "203.252.112.19"
port_num = "8888"

if __name__ == "__main__":
    k2e_model, e2k_model, k2e_trg_inv_dict, e2k_trg_inv_dict, PN_list = setting()
    app.run(host=host_addr, port=port_num, threaded=True)
