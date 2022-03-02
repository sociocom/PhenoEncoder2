#!/usr/bin/env python
# -*- coding:utf-8 -*-
import MeCab
import numpy as np
import os
import pickle
from scipy import sparse
import mojimoji
from keras.callbacks import EarlyStopping
from sklearn_crfsuite import metrics
from keras.models import Model, Sequential, Input, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.core import Dropout
from keras import optimizers
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers import Dense, Activation, Flatten, TimeDistributed
from keras.layers import Bidirectional, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Masking
from keras.layers import GlobalMaxPool1D
from keras.layers import concatenate, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from keras_contrib.layers import PELU
from keras_contrib.layers import CRF


os.chdir(os.path.dirname(__file__))


"""
糖尿病、喫煙、飲酒を識別する際の形態素解析
"""
def mrph_analysis(line):
    sw = ["【症例】", "平成", "♯", "＃", "，", "％", "←", "【まとめ】", "ｎ", "・", "＜ＢＲ＞", "【結語】", "【目的】","【方法】", "【結果】", "％", "【総括】","【対象・方法】", "【結果】", "【目的】", "【", "】", "症例;"]

    for x in sw:
        line = line.replace(x, "")

    line = str(line).replace("ﾘｽｸﾌｧｸﾀ-", "危険因子").replace("riskfactor", "危険因子").replace("冠危険因子", "危険因子").replace("無い", "ない").replace("無し", "ない").replace(",", ",")

    token = MeCab.Tagger("-u ./source/sample.dic")
    #token = MeCab.Tagger("-Ochasen")
    tokens = token.parse(line)
    tmp_pos = []
    tmp_icd = []
    tmp1 = []
    for x in tokens.split("\n"):

        if x == "EOS":
            break
        else:
            tmp_pos.append(x.split(",")[0].split("\t")[1])
            if str(x).find("icd=") != -1:
                tmp1.append(x.split("\t")[0])#(x.split("\t")[1].split(",")[-2])
            else:
                
                if str(x).find("icd=") != -1:
                    tmp_icd.append(x.split(";")[1].split(";")[0].replace("icd=", ""))
                else:
                    tmp_icd.append("None")

                if (x.split("\t")[1].split(",")[-3]) == "*":
                    tmp1.append(x.split("\t")[0])
                else:
                    tmp1.append(x.split("\t")[1].split(",")[-3])
        wakati1 = tmp1
    #stopwords = ["､", "／", "例", "た", "mg", "ｄｌ", "×", ":", "=", "･", "｢", "｣", "<", " BR", ">", "#", ")", "(", "%", ":", ",#", ",# %", "､", "(#", "),#", "､%", ")#", " ", "°(", "°", "."";", "~", ").", "%,#", "､"]
    stopwords = []
    tmp = []
    for y in wakati1:
        if y not in stopwords:
            tmp.append(y)
        else:
            pass
    return (" ".join(tmp), tmp_pos, tmp_icd)

"""
ルールベースの情報抽出
"""
class rule_base:

    def weight_extraction(self, line):
        tmp = list(line)
        tmp_weight = []
        flag = 0
        for v in range(0, len(tmp), 1):
            try:
                if (tmp[v] == "体") and (tmp[v + 1] == "重"):
                    for c in range(v, v + 9, 1):
                        if str(tmp[c]) in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0",  ".", "k", "g", "K", "G"]:
                            tmp_weight.append(tmp[c])
                            flag = 1
                        else:
                            pass
                if flag == 1:
                    break
            except:
                pass

        if len(tmp_weight) > 2:
            weight = ("".join(tmp_weight))
        else:
            weight = "-"
        return weight

    def height_extraction(self, line):
        tmp = list(line)
        tmp_weight = []
        flag = 0
        for v in range(0, len(tmp), 1):
            if (tmp[v] == "身") and (tmp[v + 1] == "長"):
                try:
                    for c in range(v, v + 8, 1):
                        if str(tmp[c]) in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0",  ".", "c", "m"]:
                            tmp_weight.append(tmp[c])
                            flag = 1
                        else:
                            pass
                except:
                    pass
            if flag == 1:
                break
        if len(tmp_weight) > 2:
            weight = ("".join(tmp_weight))
        else:
            weight = "-"
        return weight

    def HBA1C_extraction(self, line):
        tmp = list(line)
        hba1c = []
        flag = 0
        for v in range(0, len(tmp), 1):
            tmp_weight = []
            if (tmp[v] == "H") and (tmp[v + 1] == "b") and (tmp[v + 2] == "A") and (tmp[v + 3] == "1") and (tmp[v + 4] == "c"):
                for c in range(v + 5, v + 20, 1): #5, 11
                    if str(tmp[c]) in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0",  "%", ".", "-"]:
                        tmp_weight.append(tmp[c])
                        flag = 1
                    else:
                        pass
                hba1c.append(("".join(tmp_weight)))
            #if flag == 1:
                #break
                #pass
        if len(hba1c) != 0:
            tmp = []
            for w in hba1c:
                if w.find("%") != -1:
                    tmp.append(w)
                else:
                    pass
            weight = (",".join(tmp))
        else:
            weight = "-"
        return weight


    def age_extraction(self, line):
        tmp = list(line)
        tmp_weight = []
        flag = 0
        for v in range(0, len(tmp), 1):
            if (tmp[v] == "歳"):# or (tmp[v] == "代"):
                for c in range(v - 2, v + 2, 1):
                    if str(tmp[c]) in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0", "歳", "代", "五", "六", "七", "八", "九", "十"]:
                        tmp_weight.append(tmp[c])
                        flag = 1
                    else:
                        pass
            if flag == 1:
                break
        if len(tmp_weight) > 0:
            weight = ("".join(tmp_weight))
        else:
            weight = "-"
        return weight

    def CRP_extraction(self, line):
        tmp = list(line.replace(" ", ""))
        tmp_weight = []
        flag = 0
        for v in range(0, len(tmp), 1):
            if (tmp[v] == "C") and (tmp[v + 1] == "R") and (tmp[v + 2] == "P") and (tmp[v + 3] in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0",  "%", ".", "-", "m", "g", "/", "d", "l"]):
                for c in range(v + 3, v + 15, 1):
                    if str(tmp[c]) in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0",  "%", ".", "-", "m", "g", "/", "d", "l"]:
                        tmp_weight.append(tmp[c])
                        flag = 1
                    else:
                        pass
            if flag == 1:
                break
        if len(tmp_weight) > 2:
            weight = ("".join(tmp_weight))
        else:
            weight = "-"
        return weight

    def blood_pressure(self, line):
        tmp = list(line.replace(" ", ""))
        tmp_weight = []
        flag = 0
        for v in range(0, len(tmp), 1):
            if (tmp[v] == "血") and (tmp[v + 1] == "圧") and (tmp[v - 1] != "高"):
                for c in range(v, v + 14, 1):
                    try:
                        if str(tmp[c]) in ["1", "2", "3", "4",  "5", "6", "7",  "8", "9", "0",  "%", ".", "-", "m", "g", "/", "d", "l", "H"]:
                            tmp_weight.append(tmp[c])
                            flag = 1
                        else:
                            pass
                    except:
                        pass
            if flag == 1:
                break
        if len(tmp_weight) > 2:
            weight = ("".join(tmp_weight))
        else:
            weight = "-"

        try:
            tmp = weight.split("/")
            tmp_bp1 = float(tmp[0])
            tmp_bp2 = float(str(tmp[1]).replace("mmHg", ""))

            if tmp_bp1 > tmp_bp2:
                bp1 = int(tmp_bp1)
                bp2 = int(tmp_bp2)
            else:
                bp1 = int(tmp_bp2)
                bp2 = int(tmp_bp1)

        except Exception as e:
            bp1 = "-"
            bp2 = "-"

        return bp1, bp2

    def sex(self, line):
        if line.find("男性") != -1 or line.find("男") != -1:
            tmp = "男性"
        elif line.find("女性") != -1 or line.find("女") != -1:
            tmp = "女性"
        else:
            tmp = "-"
        return tmp

    def medicine_name_extraction(self, line):
        for x in ["抗血小板薬", "アスピリン", "イコサペント", "クロピドグレル", "サルポグレラート", "チクロビジン", "シロスタゾール","ジピリダモール", "チカグレロル", "ベラプロストNa（ナトリウム）"]:
            if line.find(mojimoji.zen_to_han(x)) != -1:
                tmp1 = "P"
                break
            else:
                tmp1 = "N"
        for x in ["抗凝固薬", "ワルファリン", "アセノクマロール", "フェニンジオン", "エドキサバン", "アピキサバン", "アルガトロバン","ダビガトラン", "アンチトロンビン", "リバーロキサバン", "フォンダパリヌクス", "クエン酸ナトリウム水和物", "ダビガトランエテキシラートメタンスルホン酸塩","トロンボモデュリンアルファ", "ヘパリン"]:
            if line.find(mojimoji.zen_to_han(x)) != -1:
                tmp2 = "P"
                break
            else:
                tmp2 = "N"
        for x in ["スタチン", "プラバスタチン", "シンバスタチン", "フルバスタチン","アトルバスタチン", "ピタバスタチン", "ロスバスタチン"]:
            if line.find(mojimoji.zen_to_han(x)) != -1:
                tmp3 = "P"
                break
            else:
                tmp3 = "N"
        for x in ["糖尿病治療薬", "シタグリプチン", "ビルダグリプチン", "アログリプチン", "リナグリプチン", "テネリグリプチン", "アナグリプチン"]:
            if line.find(mojimoji.zen_to_han(x)) != -1:
                tmp4 = "P"
                break
            else:
                tmp4 = "N"
        return tmp1, tmp2, tmp3, tmp4
    
    def temperature_extraction(self, line):

        tmp_char = list(line)
        tmp_value = 0
        tmp_tem = []

        for i, c in enumerate(tmp_char):
            if c == "℃":
                tmp_value = i
            else:
                pass

        if tmp_value != 0:
            for i in range(tmp_value - 5, tmp_value + 1, 1):
                if tmp_char[i] in ["1", "2", "3", "4", "5", "6", "7",  "8", "9", "0", "."]:
                    tmp_tem.append(tmp_char[i])
                else:
                    pass

            if len(tmp_tem) != 0:
                tem = "".join(tmp_tem)
            else:
                tem = "-"
        else:
            tem = "-"

        return tem
    

    def beat_extraction(self, line):
        
        tmp_char = list(line)
        tmp_value = 0
        tmp_beat = []
        
        for i, c in enumerate(tmp_char):
            if c == "b" and tmp_char[i + 1] == "p" and tmp_char[i + 2] == "m" or c == "B" and tmp_char[i + 1] == "P" and tmp_char[i + 2] == "M":
                tmp_value = i
            else:
                pass
    
        if tmp_value != 0:
            for i in range(tmp_value - 5, tmp_value + 1, 1):
                if tmp_char[i] in ["1", "2", "3", "4", "5", "6", "7",  "8", "9", "0", "."]:
                    tmp_beat.append(tmp_char[i])
                else:
                    pass
        
            if len(tmp_beat) != 0:
                beat = "".join(tmp_beat)
            else:
                beat = "-"
        else:
            beat = "-"

        return beat


"""
糖尿病、喫煙、飲酒の識別クラス
"""
class ML_base:
    def bow_vector(self, line):
        word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1])  for line in open("./source/SVM_BoW_vocab.txt", encoding = "utf-8")}
        bow = np.zeros((1, len(word2idx)))
        for j in line.split(" "):
            if j in word2idx:
                bow[:, word2idx[j]] += 1
            else:
                pass
        bow = sparse.csr_matrix(bow)
        return bow

    def bow_vector_diabetes(self, line):
        word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1])  for line in open("./source/SVM_diabetes_vocab_1011.txt", encoding = "utf-8")}
        bow = np.zeros((1, len(word2idx)))
        for j in line.split(" "):
            if j in word2idx:
                bow[:, word2idx[j]] += 1
            else:
                pass
        return bow

    def bow_vector_smoke(self, line):
        word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1])  for line in open("./source/SVM_smoke_vocab_1011.txt", encoding = "utf-8")}
        bow = np.zeros((1, len(word2idx)))
        for j in line.split(" "):
            if j in word2idx:
                bow[:, word2idx[j]] += 1
            else:
                pass
        return bow

    def smoke_prediction(self, bow):
        result = model2.predict(bow)
        return result

    def diabetes_prediction(self, bow):

        result = model1.predict(bow)
        return result

    def drink_prediction(self, bow):
            pass


"""
窓サイズを考慮したBoWベクトルの構築
"""
def ws_extraction(wakati, ws1):
    ws1 = ws1
    ws2 = 10

    tmp = []

    words = wakati.split(" ")
    lengt = len(words) - 1

    key_pos = 0
    for i, word in enumerate(words):
        if word.find("糖尿") != -1:
            key_pos = i
            break
        else:
            pass

    #前方向に何単語抽出できるか
    num_forword = ws1 - key_pos
    #後ろ
    num_backword = ws2 - key_pos

    #前方向に何単語抽出できるか
    if key_pos > ws1:
        start_forword = key_pos - ws1
    else:
        start_forword = 0

    #後ろ方向
    tmp_backward = lengt - key_pos

    if tmp_backward > ws2:
        end_backward = ws2
    else:
        end_backward = tmp_backward

    for v in range(start_forword, key_pos + 1, 1):
        tmp.append(words[v])

    for v in range(key_pos + 1,  key_pos + end_backward + 1, 1):
        tmp.append(words[v])

    return (" ".join(tmp))


"""
病名抽出
"""

def information_extraxtion(wakati, pos, icd):

    word_length = 30
    char_length = 10

    wakati = wakati.split(" ")

    idx2word = {int(line.replace("\n", "").split("\t")[1]): line.split("\t")[0] for line in open("./source/BiLSTM_ER_word_1115.txt", encoding="utf-8")}
    word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("./source/BiLSTM_ER_word_1115.txt", encoding="utf-8")}
    char2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("./source/BiLSTM_ER_char_1115.txt", encoding="utf-8")}
    pos2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("./source/BiLSTM_ER_POS_1115.txt", encoding="utf-8")}
    #icd2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("./source/BiLSTM_ER_ICD_1123.txt", encoding="utf-8")}

    """
    品詞ベクトルの構築
    """
    X_TMP = [[pos2idx[z] for z in (" ".join(pos)).split(" ")]]
    
    X_POS = pad_sequences(maxlen = word_length, sequences=X_TMP, padding="post", value=0, truncating='post')
    
    #X_TMP = [[icd2idx[z] for z in (" ".join(icd)).split(" ")]]
    
    #X_ICD = pad_sequences(maxlen = word_length, sequences=X_TMP, padding="post", value=0, truncating='post')

    """
    文字ベクトルの構築
    """
    X_CHAR = []
    sent_seq = []
    ws = str(wakati).split(" ")
    for i in range(word_length):
        word_seq = []
        for  j in range(char_length):
            try:
                tmp = list(ws[i])[j]

                if tmp in char2idx:
                    word_seq.append(char2idx[tmp])
                else:
                    word_seq.append(char2idx["UNK"])

            except:
                word_seq.append(char2idx["PAD"])


        sent_seq.append(word_seq)
    X_CHAR.append(np.array(sent_seq))


    """
    形態素ベクトルの構築
    """
    X = []
    for b in wakati:
        if b in word2idx:
            X.append(word2idx[b])
        else:
            X.append(word2idx["UNK"])
    X = [X]
    X_MRPH = pad_sequences(maxlen = word_length, sequences = X, padding="post", value = 0, truncating='post')


    #predictions = (model_ER.predict([np.array(X_MRPH), np.array(X_POS), np.array(X_CHAR)]))
    predictions = (model_ER.predict([np.array(X_MRPH), np.array(X_POS),np.array(X_CHAR)]))

#id2tag = {1: "nan", 2: "I-P", 3: "O", 4: "B-N", 5: "I-N", 6: "B-P", 0: "Padding"} 1109
    id2tag = {1: "B-P", 2: "B-N", 3: "O", 4: "I-N", 5: "I-P", 0: "Padding"} #1115
    #id2tag = {1: "O", 2: "B-P", 3: "B-N", 4:"I-P", 5: "I-N", 0:"Padding"} #1126

    ptags = []
    ntags = []

    words = [idx2word[n] for n in X_MRPH[0]]
    proba = np.argmax(predictions, axis=-1)

    for i, (x, y) in enumerate(zip(proba[0], words)):
        try:
            #print(str(x) + "      " + str(y) + "\n")
            tmp = []
            if id2tag[x] != "Padding" and y != "":
                if id2tag[x] == "B-P":
                    tmp.append(words[i])
                    for c in range(i + 1, i + 20, 1):
                        if id2tag[proba[0][c]] == "I-P":
                            tmp.append(words[c])

                        else:
                            break
                    ptags.append("".join(tmp))

                elif id2tag[x] == "B-N":
                    tmp.append(words[i])
                    for c in range(i + 1, i + 20, 1):
                        if id2tag[proba[0][c]] == "I-N":
                            tmp.append(words[c])
                        else:
                            break
                    ntags.append("".join(tmp))

                else:
                    pass

            else:
                pass
        except:
            pass
    return ptags, ntags

def start( texts, PEmode ):

    ML = ML_base()
    #モデル読み込み
    global model1, model2, model_ER
    if PEmode == "A1" :
        model1 = pickle.load(open("./source/SVM_diabetes_1011.sav", 'rb'))
        model2 = pickle.load(open("./source/SVM_smoke_1011.sav", 'rb'))
        model_ER = load_model("./source/model_1115_wc.h5", custom_objects={'CRF': CRF})
    elif PEmode == "B1":
        model1 = pickle.load(open("./source/SVM_diabetes_1011.sav", 'rb'))
        model2 = pickle.load(open("./source/SVM_smoke_1011.sav", 'rb'))
    else:
        model_ER = load_model("./source/model_1115_wc.h5", custom_objects={'CRF': CRF})
    
    # htmlタグを埋め込んだ症例結果を保存する配列
    syoureigun = []
    csv_data = []

    for line in texts:
        line = line.replace('\t', '')
        # 入力テキストがない場合は処理をスキップ
        if(line==''):
            continue

        line = mojimoji.zen_to_han(line)
        line = line.replace(" ", "").replace(",", "・")
    

        """
        変数定義
        """
        all_ptags = []
        sent = []
        #最終要約用の辞書
        sum_dict = {}
        for c in ["weight", "hight", "hba1c", "age", "crp", "maxp", "minp", "m1", "m2", "m3", "m4", "sex", "diabetes", "smokes", "drinks", "tem", "beat"]:
            sum_dict[c] = []

        for i, x in enumerate([x for x in line.split("｡") if x != ""]):

            if x == "" or x =="\r\n":
                break
            else:
                pass

            wakati, pos, icd = mrph_analysis(x)

            """
            病名抽出の実行
            """
            if PEmode == "A1" or PEmode == "C1":

                ptags, ntags = information_extraxtion(wakati, pos, icd)

                """
                陽性病名は赤、陰性病名は青に
                """
                if len(ptags) != 0:
                    for w in set(ptags):
                        w = w.replace("UNK", "")
                        if w != "":
                            all_ptags.append(w)
                            x = x.replace(w, "<font size=\"2\" color=\"#ff0000\">" + w + "</font>")

                if len(ntags) != 0:
                    for ww in set(ntags):
                        ww = ww.replace("UNK", "")
                        if ww != "":
                            x = x.replace(ww, "<font size=\"2\" color=\"blue\">" + ww + "</font>")

                sent.append(x)

            else:
                pass

            """
            機械学習・ルールベースの実行
            """
            rule = rule_base()
            weight = rule.weight_extraction(x)
            hight = rule.height_extraction(x)
            hba1c = rule.HBA1C_extraction(x)
            age = rule.age_extraction(x)
            crp = rule.CRP_extraction(x)
            maxp = rule.blood_pressure(x)[0]
            minp = rule.blood_pressure(x)[1]
            m1 = rule.medicine_name_extraction(x)[0]
            m2 = rule.medicine_name_extraction(x)[1]
            m3 = rule.medicine_name_extraction(x)[2]
            m4 = rule.medicine_name_extraction(x)[3]
            sex = rule.sex(x)
            tem = rule.temperature_extraction(x)
            beat = rule.beat_extraction(x)

            #最終結果用
            sum_dict["weight"].append(weight)
            sum_dict["hight"].append(hight)
            sum_dict["hba1c"].append(hba1c)
            sum_dict["age"].append(age)
            sum_dict["crp"].append(crp)
            sum_dict["maxp"].append((maxp))
            sum_dict["minp"].append((minp))
            sum_dict["m1"].append(m1)
            sum_dict["m2"].append(m2)
            sum_dict["m3"].append(m3)
            sum_dict["m4"].append(m4)
            sum_dict["sex"].append(sex)
            sum_dict["tem"].append(tem)
            sum_dict["beat"].append(beat)

            """
            糖尿病に関する処理
            """
        
            if PEmode != "C1":
    
                if x.find("糖尿") != -1:
                    wakati = ws_extraction(wakati, 0)
                    bow_diabetes = ML.bow_vector_diabetes(wakati)
                    diabetes_result = ML.diabetes_prediction(bow_diabetes)
                    diabetes = diabetes_result[0]
                    if (diabetes) == 0:
                        diabetes = "N"
                    else:
                        diabetes = "P"
                    sum_dict["diabetes"].append(diabetes)
                else:
                    sum_dict["diabetes"].append("U")
                    diabetes = "U"

            else:
                pass

            """
            喫煙
            """
            if PEmode != "C1":
                if x.find("喫煙") != -1 or x.find("禁煙") != -1 or x.find("タバコ") != -1 or x.find("ﾀﾊﾞｺ") != -1 or x.find("嗜好") != -1 or x.find("煙草") != -1:
                    wakati = ws_extraction(wakati, 5)
                    bow_smoke = ML.bow_vector_smoke(wakati)
                    smoke_result = ML.smoke_prediction(bow_smoke)
                    smoke = smoke_result[0]
                    if (smoke) == 1:
                        smoke = "N"
                    elif smoke == 2:
                        smoke = "P1"
                    else:
                        smoke = "P2"
                    sum_dict["smokes"].append(smoke)
                else:
                    sum_dict["smokes"].append("U")
                    smoke = "U"

            """
            飲酒
            """
            drinks = 0
            if drinks == 0:
                drinks = "U"
            elif drinks == 1:
                drinks = "N"
            elif drinks == 2:
                drinks = "P1"
            elif drinks == 3:
                drinks = "P2"
            else:
                drinks = "P3"
            sum_dict["drinks"].append(drinks)


        #最終出力用辞書の整理
        dict = {}
        for x, y in sum_dict.items():
            tmp = [str(z) for z in y if z != "-"]
            if len(tmp) == 0:
                dict[x] = "-"
            else:
                #P-UとかではUを表示しないようにする
                if (list(set(tmp)))[0] == "U" and len(list(set(tmp))) == 1:
                    dict[x] = (" ".join(list(set(tmp))))
                else:
                    dict[x] = (" ".join(list(set(tmp)))).replace("U", "")

        #薬品をP or Nの一つに
        for c in ["m1", "m2", "m3", "m4"]:
            if dict[c] == "P N":
                dict[c] = "P"
            elif dict[c] == "N P":
                dict[c] = "P"
            elif dict[c] == "P":
                dict[c] = "P"
            else:
                dict[c] = "N"


        """
        診断名の確定
        """
        if PEmode == "A1" or PEmode == "C1":
            char_list = list(line)
            tmp_value = 0
            for i , w in enumerate(char_list):
                if char_list[i] == "診" and char_list[i + 1] == "断" or char_list[i] == "確" and char_list[i + 1] == "断" or char_list[i] == "判" and char_list[i + 1] == "明":
                    tmp1 = mojimoji.han_to_zen("".join(char_list[i - 12 : i]))
                    for c in all_ptags:
                        #print("{0}: {1}".format(c, tmp1))
                        #print("\n")
                        c = c.replace("UNK", "")
                        if tmp1.find(mojimoji.han_to_zen(c)) != -1:
                            diagnosis_name = mojimoji.han_to_zen(c)
                            #print(c + "\n")
                        #break
                        else:
                            #diagnosis_name = "-"
                            pass
            try:
                dict["診断名"] = diagnosis_name
            except:
                dict["診断名"] = "-"
        else:
            dict["診断名"] = "-"
    
        """
        要約を実行しない場合、全て削除
        """
        if PEmode == "C1":
            for c in ["weight", "hight", "hba1c", "age", "crp", "maxp", "minp", "m1", "m2", "m3", "m4", "sex", "diabetes", "smokes", "drinks", "tem", "beat"]:
                dict[c] = "-"
        

        # 要約結果群を配列に格納
        csv_data.append( [ line, dict["sex"], dict["hight"], dict["weight"], dict["age"], dict["hba1c"], dict["crp"], dict["maxp"]+'-'+dict["minp"]+'mmHg', dict["tem"]+'℃', dict["beat"]+'bpm', dict["m1"], dict["m2"], dict["m3"], dict["m4"],dict["diabetes"], dict["smokes"], dict["drinks"], dict["診断名"]] )
        # 各入力テキストにhtmlタグを埋め込んだ病名抽出の結果を保存
        syoureigun.append(mojimoji.han_to_zen("。".join(sent), digit=False, ascii=False))
    
    return syoureigun, csv_data




"""
メイン関数
"""
if __name__ == "__main__":
    
    text = ['Liddle症候群が疑われたが、偽性アルドステロン症と鑑別された症例を報告する。症例は81歳の女性で、体重70Kｇ、身長157cm。病前は独步でADLは自立。既往歴には高脂血症、前壁梗塞のため冠動脈バイパス術、2型糖尿病のためインスリン治療を受けていた。家族歴は従姉妹：胃癌、低カリウム血症の家族歴は明らかでない。生活歴：4か月前より禁煙。現病歴はX年２月頃、近医にて下肢有痛性筋痙攣のため芍薬甘草湯の内服を開始した。同年３月より全身倦怠感と夜間呼吸困難を認めるようになり、買い物中に意識朦朧となり、当院へ救急搬送され緊急入院となった。入院時の体温37．3℃、血圧141／80　mmHg、脈拍68bpm、呼吸数18回／分、SpO298％（room air）。血液検査ではカリウム1．6　mEqA、レニン活性低下、心電図変化を認め、偽性アルドステロン症と診断し、保存的加療を行った。入院後、芍薬甘草湯内服は中止、心不全薬物療法、カリウム製剤補充を行った。血清カリウム値の是正に難渋し、入院中に心不全再増悪を認め、甘草湯の薬効残存が考えられた。']

    syourei, csv_data = start( text, 'A1' )
    print(csv_data)