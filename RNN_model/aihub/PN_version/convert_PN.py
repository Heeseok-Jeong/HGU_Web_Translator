import six; from six.moves import cPickle as pkl

#1. src_file을 읽고 인물 보캡에 있는 이름은 순서대로 가림 __P0, __P1 ...
#2. trg_file도 마찬가지, 여기까진 쉬움
#3. 중요한건 trgid 할 때 각각 __P0에 관한 정보(P0:John, 요한)을 기억하다가 모델을 통해 나온 센텐스에 끼워넣어주면 됨.

if __name__ == "__main__":
    src_file_name1 = "./aihub_train.kr.shuf.tok"
    src_file_name2 = "./aihub_train.en.shuf.tok"
    trg_file_name1 = "./aihub_train.kr.shuf.tok.pn"
    trg_file_name2 = "./aihub_train.en.shuf.tok.tok.pn"
    dict_file_name = "./vocab/extractPN.pkl"

    src_file1 = open(src_file_name1, "r", encoding="utf-8")
    src_file2 = open(src_file_name2, "r", encoding="utf-8")
    with open(dict_file_name, 'rb') as f: #with 구문은 따로 file close를 안해도 자동으로 close를
        PN_dict = pkl.load(f, encoding="utf-8") #직렬화되어있는 파일을 피클로 로드하여 역직렬화 진행
    # print(PN_dict)

    temp_src1 = ""
    temp_src2 = ""
    src_line1 = src_file1.readline()
    src_line2 = src_file2.readline()

    while src_line1:
        i = 0
        for key, val in PN_dict.items(): #key는 한국어, val은 영어
            temp = " __P" + str(i) + " "
            if key in src_line1:
                if val in src_line2:
                    # print(val)
                    src_line1 = src_line1.replace(key, temp)
                    src_line2 = src_line2.replace(val, temp)
                    i = i + 1
                # print(key)
        temp_src1 = temp_src1 + src_line1
        temp_src2 = temp_src2 + src_line2
        src_line1 = src_file1.readline()
        src_line2 = src_file2.readline()

    with open(trg_file_name1, "w")as nf1:
        nf1.write(temp_src1)
    with open(trg_file_name2, "w")as nf2:
        nf2.write(temp_src2)


def convert_pn_for_web(input_sen, PN_list, info_dict, lang):
    # with open(dict_file_name, 'rb') as f:
    #     PN_dict = pkl.load(f, encoding="utf-8")
    # print(PN_dict)
    i = 0
    print(PN_list[0])
    print(PN_list[0][0])
    print(PN_list[0][1])

    for (key, val) in PN_list :
        key = key.strip()
        val = val.strip()
        temp = " __P" + str(i) + " "
        if lang == "k2e": #한->영
            if key in input_sen:
                input_sen = input_sen.replace(key, temp)
                info_dict[temp] = key #example => info_dict[ __P0 ] = 예수
                i = i + 1
        else: #영->한
            if val in input_sen:
                input_sen = input_sen.replace(val, temp)
                info_dict[temp] = val
                i = i + 1

    return input_sen, info_dict

    # for key, val in PN_dict.items(): #key는 한국어, val은 영어
    #     key = key.strip()
    #     val = val.strip()
    #     temp = " __P" + str(i) + " "
    #     if lang == "k2e": #한->영
    #         if key in input_sen:
    #             input_sen = input_sen.replace(key, temp)
    #             info_dict[temp] = key #example => info_dict[ __P0 ] = 예수
    #             i = i + 1
    #     else: #영->한
    #         if val in input_sen:
    #             input_sen = input_sen.replace(val, temp)
    #             info_dict[temp] = val
    #             i = i + 1
    #
    # return input_sen, info_dict
