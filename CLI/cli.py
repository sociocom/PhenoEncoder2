import PE
import argparse
import csv
import re

"""
メイン関数
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group()

    group1.add_argument("-t", "--inputText", help="テキスト入力", nargs=1, default='')
    group1.add_argument("-i", "--inputFile", help="入力ファイルのパス", nargs=1, default='')
    parser.add_argument("-p", "--phenoType", help="機能選択．0(default): 全機能, 1: 要約のみ, 2: 病名抽出のみ ", nargs=1, choices=['0', '1', '2'],default='0')
    parser.add_argument("-o", "--outputFile", help="出力先ファイルのパス", nargs=1, default='output.csv')

    # namespaceオブジェクトの生成
    args = parser.parse_args()
    text = []

    #排他グループによる処理分岐
    if args.inputText != '':
        value="".join(args.inputText)
        text = [value]

    #--addを設定した場合の処理
    elif args.inputFile != '':
        value = []
        csv_file = open(args.inputFile[0], "r")
        f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        for row in f:
            value.append(row[0])
        csv_file.close()
        text = value

    #排他グループの引数が設定されない場合の処理
    else:
        value = '''Liddle症候群が疑われたが、偽性アルドステロン症と鑑別された症例を報告する。症例は81歳の女性で、体重70Kｇ、身長157cm。病前は独步でADLは自立。既往歴には高脂血症、前壁梗塞のため冠動脈バイパス術、2型糖尿病のためインスリン治療を受けていた。家族歴は従姉妹：胃癌、低カリウム血症の家族歴は明らかでない。生活歴：4か月前より禁煙。現病歴はX年２月頃、近医にて下肢有痛性筋痙攣のため芍薬甘草湯の内服を開始した。同年３月より全身倦怠感と夜間呼吸困難を認めるようになり、買い物中に意識朦朧となり、当院へ救急搬送され緊急入院となった。入院時の体温37．3℃、血圧141／80　mmHg、脈拍68bpm、呼吸数18回／分、SpO298％（room air）。血液検査ではカリウム1．6　mEqA、レニン活性低下、心電図変化を認め、偽性アルドステロン症と診断し、保存的加療を行った。入院後、芍薬甘草湯内服は中止、心不全薬物療法、カリウム製剤補充を行った。血清カリウム値の是正に難渋し、入院中に心不全再増悪を認め、甘草湯の薬効残存が考えられた。'''
        text = [value]
    type1 = 'A1'
    if str(args.phenoType[0]) == '1':
        type1 = 'B1'
    elif str(args.phenoType[0]) == '2':
        type1 = 'C1'
    else:
        pass


    syoureigun, csv_data = PE.start(text, type1)

    # csvに書き出し
    if args.outputFile == 'output.csv':
        output_file_name = args.outputFile
    else:
        output_file_name = str(args.outputFile[0])
    with open(output_file_name, 'w') as f:
        writer = csv.writer ( f )
        for i, data in enumerate(csv_data):
            writer.writerow( ['原文', '性別', '身長', '体重', '年齢', 'HbA1c', 'CRP', '血圧', '体温', '脈拍', '抗血小板薬', '抗凝固薬', 'スタチン', '糖尿病治療薬', '糖尿病', '喫煙', '飲酒', '診断名', '陽性所見', '陰性所見'])
            if type1 != 'B1':
                csv_row = data.extend(" ".join( re.findall(r'<font size=\"2\" color=\"#ff0000\">(.+?)</font>', syoureigun[i])))
                csv_row = data.extend(" ".join( re.findall(r'<font size=\"2\" color=\"blue\">(.+?)</font>', syoureigun[i])))
                writer.writerow( csv_row )
            else:
                writer.writerow( data )
