#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cgi
import cgitb
import csv
import io
import sys
from tool import PE

"""
メイン関数
"""
if __name__ == "__main__":
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    cgitb.enable()

    # パラメーターを取得
    form = cgi.FieldStorage()
    #チェックボックスのパラメータを取得
    type1 = form.getvalue('riyu')

    # テキストボックスの文字列を取得    
    # jsプログラムを読み込む場合は以下を使用する（複製されたフォームごとに実行されるようになる）
    # value = form.getlist("form_text")
    value = str(form['form_text'].value)
    value = value.split('\r\n')


    syoureigun, csv_data = PE.start(value, type1)
    
    os.chdir(os.path.dirname(__file__))

    # csvに書き出し
    # with open('./output/result.csv', 'w') as f:
    #     writer = csv.writer ( f )
    #     writer.writerow( ['原文', '性別', '身長', '体重', '年齢', 'HbA1c', 'CRP', '血圧', '体温', '脈拍', '抗血小板薬', '抗凝固薬', 'スタチン', '糖尿病治療薬', '糖尿病', '喫煙', '飲酒', '診断名'])
    #     for data in csv_data:
    #         writer.writerow( data )

    """
    HTMLの表示
    """
    print("Content-type: text/html; charset=UTF-8\r\n")
    print("<title>Pheno Encoder</title>")
    print("<h3>要約結果</h3><style>h3 {color: #364e96;padding: 0.5em 0;border-top: solid 3px #364e96;border-bottom: solid 3px #364e96;text-align:center}</style>")
    
    for data in csv_data:
        print("<CENTER><table border=\"1\"><tr width=100%><tr><th>性別</th><th>身長</th><th>体重</th><th>年齢</th><th>HbA1c</th><th>CRP</th><th>血圧</th><th>体温</th><th>脈拍</th><th>抗血小板薬</th><th>抗凝固薬</th><th>スタチン</th><th>糖尿病治療薬</th><th>糖尿病</th><th>喫煙</th><th>飲酒</th><th>診断名</th></tr><tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}℃</td><td>{8}bpm</td><td>{9}</td><td>{10}</td><td>{11}</td><td>{12}</td><td>{13}</td><td>{14}</td><td>{15}</td><td>{16}</td></tr></table></CENTER>".format(data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17]))
        print("<BR>")
    
    print("<ul><li><font size=\"2\" color=\"#000000\">抗血小板薬，抗凝固薬，スタチン，糖尿病治療薬: P(投与あり), N(投与なし)</font></li><li><font size=\"2\" color=\"#000000\">糖尿病: P(糖尿病あり), N(糖尿病なし), U (記載なし)</font></li><li><font size=\"2\" color=\"#000000\">喫煙: P1 (喫煙あり), P2 (過去の喫煙あり), N(喫煙歴なし), U (記載なし)</font></li><li><font size=\"2\" color=\"#000000\">飲酒: P1 (飲酒あり), P2 (過去の飲酒あり), N(飲酒歴なし), U (記載なし): <font size=\"2\" color=\"#ff0000\">現在停止中</font></font></li></ul>")
    print("<h3>病名抽出</h3>")
    for syourei in syoureigun:
        print("<div style=\"padding: 10px; margin-bottom: 10px; border: 1px solid #333333; border-radius: 10px;\"><font size=\"2\" color=\"#000000\">{0}</font></div>".format(syourei))
    print("<ul><li><font size=\"2\" color=\"#000000\">赤色の単語は陽性所見，青色の単語は陰性所見を示す．</font></li></ul>")
    print('<button type="button" onClick="history.back()">戻る</button>')
    