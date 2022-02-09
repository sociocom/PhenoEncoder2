#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cgi
import cgitb
import MeCab
import numpy as np
import os
import io
import sys
import pickle
from scipy import sparse
import mojimoji
from keras.callbacks import EarlyStopping
from sklearn_crfsuite import metrics
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Input
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras import optimizers
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Bidirectional
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import  BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Masking
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, TimeDistributed
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from keras.models import load_model
from keras_contrib.layers import PELU
from keras_contrib.layers import CRF
import csv


os.chdir(os.path.dirname(__file__))


"""
糖尿病、喫煙、飲酒を識別する際の形態素解析
"""
def mrph_analysis(line):
    sw = ["【症例】", "平成", "♯", "＃", "，", "％", "←", "【まとめ】", "ｎ", "・", "＜ＢＲ＞", "【結語】", "【目的】","【方法】", "【結果】", "％", "【総括】","【対象・方法】", "【結果】", "【目的】", "【", "】", "症例;"]

    for x in sw:
        line = line.replace(x, "")

    line = str(line).replace("ﾘｽｸﾌｧｸﾀ-", "危険因子").replace("riskfactor", "危険因子").replace("冠危険因子", "危険因子").replace("無い", "ない").replace("無し", "ない").replace(",", ",")

    token = MeCab.Tagger("-u /usr/local/lib/mecab/dic/userdic/sample.dic")
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
        word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1])  for line in open("SVM_BoW_vocab.txt", encoding = "utf-8")}
        bow = np.zeros((1, len(word2idx)))
        for j in line.split(" "):
            if j in word2idx:
                bow[:, word2idx[j]] += 1
            else:
                pass
        bow = sparse.csr_matrix(bow)
        return bow

    def bow_vector_diabetes(self, line):
        word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1])  for line in open("SVM_diabetes_vocab_1011.txt", encoding = "utf-8")}
        bow = np.zeros((1, len(word2idx)))
        for j in line.split(" "):
            if j in word2idx:
                bow[:, word2idx[j]] += 1
            else:
                pass
        return bow

    def bow_vector_smoke(self, line):
        word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1])  for line in open("SVM_smoke_vocab_1011.txt", encoding = "utf-8")}
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

    idx2word = {int(line.replace("\n", "").split("\t")[1]): line.split("\t")[0] for line in open("BiLSTM_ER_word_1115.txt", encoding="utf-8")}
    word2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("BiLSTM_ER_word_1115.txt", encoding="utf-8")}
    char2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("BiLSTM_ER_char_1115.txt", encoding="utf-8")}
    pos2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("BiLSTM_ER_POS_1115.txt", encoding="utf-8")}
    #icd2idx = {line.split("\t")[0]: int(line.replace("\n", "").split("\t")[1]) for line in open("BiLSTM_ER_ICD_1123.txt", encoding="utf-8")}

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

"""
メイン関数
"""
if __name__ == "__main__":
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # htmlを出力
    print("Content-type: text/html; charset=UTF-8\r\n")
    print("<title>Pheno Encoder</title>")
    print("<h3>要約結果</h3><style>h3 {color: #364e96;padding: 0.5em 0;border-top: solid 3px #364e96;border-bottom: solid 3px #364e96;text-align:center}</style>")

    cgitb.enable()

    # csvに書き込む際のヘッダーを設定
    with open('./sample_quote.csv', 'w') as f:
        writer = csv . writer ( f )
        writer . writerow ( ['性別', '身長', '体重', '年齢', 'HbA1c', 'CRP', '血圧', '体温', '脈拍', '抗血小板薬', '抗凝固薬', 'スタチン', '糖尿病治療薬', '糖尿病', '喫煙', '飲酒', '診断名'])

    # パラメーターを取得
    form = cgi.FieldStorage()
    #チェックボックスのパラメータを取得
    type1 = form.getvalue('riyu')
    ML = ML_base()
    #モデル読み込み
    if type1 == "A1" :
        model1 = pickle.load(open("SVM_diabetes_1011.sav", 'rb'))
        model2 = pickle.load(open("SVM_smoke_1011.sav", 'rb'))
        model_ER = load_model("model_1115_wc.h5", custom_objects={'CRF': CRF})
    elif type1 == "B1":
        model1 = pickle.load(open("SVM_diabetes_1011.sav", 'rb'))
        model2 = pickle.load(open("SVM_smoke_1011.sav", 'rb'))
    else:
        model_ER = load_model("model_1115_wc.h5", custom_objects={'CRF': CRF})

    #テキストボックスの文字列を取得
    value = str(form['form_text'].value)
    value = value.split('\r\n')
    # jsプログラムを読み込む場合は以下を使用する（複製されたフォームごとに実行されるようになる）
    #value = form.getlist("form_text")

    # htmlタグを埋め込んだ症例結果を保存する配列
    syoureigun = []
    for line in value:

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
            if type1 == "A1" or type1 == "C1":

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
        
            if type1 != "C1":
    
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
            if type1 != "C1":
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
        if type1 == "A1" or type1 == "C1":
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
        if type1 == "C1":
            for c in ["weight", "hight", "hba1c", "age", "crp", "maxp", "minp", "m1", "m2", "m3", "m4", "sex", "diabetes", "smokes", "drinks", "tem", "beat"]:
                dict[c] = "-"
        

        csv_data = [line, dict["sex"], dict["hight"], dict["weight"], dict["age"], dict["hba1c"], dict["crp"], dict["maxp"]+'-'+dict["minp"]+'mmHg', dict["tem"]+'℃', dict["beat"]+'bpm', dict["m1"], dict["m2"], dict["m3"], dict["m4"],dict["diabetes"], dict["smokes"], dict["drinks"], dict["診断名"]]
        
        # csvに各要約結果を保存
        with open('./sample_quote.csv', 'a') as f:
            writer = csv . writer ( f )
            writer . writerow ( csv_data )

        # 各入力テキストにhtmlタグを埋め込んだ病名抽出の結果を保存
        syoureigun.append(mojimoji.han_to_zen("。".join(sent), digit=False, ascii=False))
        
        
        
        """
        HTMLの表示
        """
        print("<CENTER><table border=\"1\"><tr width=100%><tr><th>性別</th><th>身長</th><th>体重</th><th>年齢</th><th>HbA1c</th><th>CRP</th><th>血圧</th><th>体温</th><th>脈拍</th><th>抗血小板薬</th><th>抗凝固薬</th><th>スタチン</th><th>糖尿病治療薬</th><th>糖尿病</th><th>喫煙</th><th>飲酒</th><th>診断名</th></tr><tr><td>{14}</td><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}-{6}mmHg</td><td>{16}℃</td><td>{17}bpm</td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td><td>{11}</td><td>{12}</td><td>{13}</td><td>{15}</td></tr></table></CENTER>".format(dict["hight"], dict["weight"], dict["age"], dict["hba1c"], dict["crp"], dict["maxp"], dict["minp"], dict["m1"], dict["m2"], dict["m3"], dict["m4"],dict["diabetes"], dict["smokes"], dict["drinks"], dict["sex"], dict["診断名"], dict["tem"], dict["beat"]))
        print("<BR>")
    
    print("<ul><li><font size=\"2\" color=\"#000000\">抗血小板薬，抗凝固薬，スタチン，糖尿病治療薬: P(投与あり), N(投与なし)</font></li><li><font size=\"2\" color=\"#000000\">糖尿病: P(糖尿病あり), N(糖尿病なし), U (記載なし)</font></li><li><font size=\"2\" color=\"#000000\">喫煙: P1 (喫煙あり), P2 (過去の喫煙あり), N(喫煙歴なし), U (記載なし)</font></li><li><font size=\"2\" color=\"#000000\">飲酒: P1 (飲酒あり), P2 (過去の飲酒あり), N(飲酒歴なし), U (記載なし): <font size=\"2\" color=\"#ff0000\">現在停止中</font></font></li></ul>")
    print("<h3>病名抽出</h3>")
    for syourei in syoureigun:
        print("<div style=\"padding: 10px; margin-bottom: 10px; border: 1px solid #333333; border-radius: 10px;\"><font size=\"2\" color=\"#000000\">{0}</font></div>".format(syourei))
    print("<ul><li><font size=\"2\" color=\"#000000\">赤色の単語は陽性所見，青色の単語は陰性所見を示す．</font></li></ul>")
    print("<h4><a href=\"http://localhost:8000\">戻る</a><h4><style>h4 {position: relative;color: #158b2b;font-size: 20px;padding: 10px 0;text-align: center;margin: 1.5em 0;}h1:before {content: "";position: absolute;top: -8px;left: 50%;width: 150px;height: 58px;border-radius: 50%;border: 5px solid #a6ddb0;border-left-color: transparent;border-right-color: transparent;-moz-transform: translateX(-50%);-webkit-transform: translateX(-50%);-ms-transform: translateX(-50%);transform: translateX(-50%);}</style>")
    