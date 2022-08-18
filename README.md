# PhenoEncoder2
## 概要
脳腫瘍の原因となる糖尿病と喫煙情報に関係する病名抽出と患者の要約作成するツールです．\
csvファイルからカルテ要約作成と病名を抽出します．

## 手法
ルールベースにより患者の要約を作成し，SVMを使用して病名の抽出します。

詳細は[こちら](https://www.jstage.jst.go.jp/article/jsaisigtwo/2018/AIMED-006/2018_01/_article/-char/ja/)の論文をご覧ください.
動作チェックを行いたい場合は[デモ環境](http://aoi.naist.jp/~shibata/PhenoEncoder/sample%20app/)にアクセスすると確認ができます．


## 環境
```
Python 3.6.9
```
（注）M1 macには対応していません．windowsの方は `NVIDIA CUDA Toolkit`及び`NVIDIA CUDA Toolkit`等ツールをインストールください（[参考](https://gist.github.com/mignonstyle/083c9e1651d7734f84c99b8cf49d57fa)）．

## ローカルでの実行方法
[MeCab](https://taku910.github.io/mecab/)のインストールを行ってください．

mac環境ではbrewでインストールできます．
Mecabのインストール後は以下のコマンドで行ってください．
(仮想環境内での実行を推奨します)
```
git clone git@github.com:sociocom/PhenoEncoder2.git
pip install -r PhenoEncoder2/requirement.txt
```

## ブラウザから実行
下記コマンドを実行後，[http://localhost:8000](http://localhost:8000)　にアクセスすると実行可能です.
- Dockerをインストール済の場合
```
docker-compose up
```
- Dockerをインストールしていない場合
```
cd webapp
python cgiserver.py
```


## CLI（ターミナル）から実行
`python CLI/cli.py` で実行可能です。

オプションの`-t`または`-i`を指定せず実行した場合、デフォルトの例文が要約対象となります。
### オプション
- `-t` or `--inputText`：カルテテキストのコマンドライン入力
- `-i` or `--inputFile`：入力ファイル名
- `-o` or `--phenoType`：出力ファイル名
- `-p` or `--outputFile`：機能選択 
  - `0`: 要約および病名抽出 (default)
  - `1`: 要約のみ
  - `2`: 病名抽出のみ

### 実行例
#### コマンド
`python CLI/cli.py -i CLI/sample.csv`
#### 入力 ( sample.csv )
```
Liddle症候群が疑われたが､偽性ｱﾙﾄﾞｽﾃﾛﾝ症と鑑別された症例を報告する｡症例は81歳の女性で､体重70Kg､身長157cm｡病前は独步でADLは自立｡既往歴には高脂血症､前壁梗塞のため冠動脈ﾊﾞｲﾊﾟｽ術､2型糖尿病のためｲﾝｽﾘﾝ治療を受けていた｡家族歴は従姉妹:胃癌､低ｶﾘｳﾑ血症の家族歴は明らかでない｡生活歴:4か月前より禁煙｡現病歴はX年2月頃､近医にて下肢有痛性筋痙攣のため芍薬甘草湯の内服を開始した｡同年3月より全身倦怠感と夜間呼吸困難を認めるようになり､買い物中に意識朦朧となり､当院へ救急搬送され緊急入院となった｡入院時の体温37.3℃､血圧141/80mmHg､脈拍68bpm､呼吸数18回/分､SpO298%(roomair)｡血液検査ではｶﾘｳﾑ1.6mEqA､ﾚﾆﾝ活性低下､心電図変化を認め､偽性ｱﾙﾄﾞｽﾃﾛﾝ症と診断し､保存的加療を行った｡入院後､芍薬甘草湯内服は中止､心不全薬物療法､ｶﾘｳﾑ製剤補充を行った｡血清ｶﾘｳﾑ値の是正に難渋し､入院中に心不全再増悪を認め､甘草湯の薬効残存が考えられた｡
```
#### 出力
```CLI/output.csv```

```
原文,性別,身長,体重,年齢,HbA1c,CRP,血圧,体温,脈拍,抗血小板薬,抗凝固薬,スタチン,糖尿病治療薬,糖尿病,喫煙,飲酒,診断名
Liddle症候群が疑われたが､偽性ｱﾙﾄﾞｽﾃﾛﾝ症と鑑別された症例を報告する｡症例は81歳の女性で､体重70Kg､身長157cm｡病前は独步でADLは自立｡既往歴には高脂血症､前壁梗塞のため冠動脈ﾊﾞｲﾊﾟｽ術､2型糖尿病のためｲﾝｽﾘﾝ治療を受けていた｡家族歴は従姉妹:胃癌､低ｶﾘｳﾑ血症の家族歴は明らかでない｡生活歴:4か月前より禁煙｡現病歴はX年2月頃､近医にて下肢有痛性筋痙攣のため芍薬甘草湯の内服を開始した｡同年3月より全身倦怠感と夜間呼吸困難を認めるようになり､買い物中に意識朦朧となり､当院へ救急搬送され緊急入院となった｡入院時の体温37.3℃､血圧141/80mmHg､脈拍68bpm､呼吸数18回/分､SpO298%(roomair)｡血液検査ではｶﾘｳﾑ1.6mEqA､ﾚﾆﾝ活性低下､心電図変化を認め､偽性ｱﾙﾄﾞｽﾃﾛﾝ症と診断し､保存的加療を行った｡入院後､芍薬甘草湯内服は中止､心不全薬物療法､ｶﾘｳﾑ製剤補充を行った｡血清ｶﾘｳﾑ値の是正に難渋し､入院中に心不全再増悪を認め､甘草湯の薬効残存が考えられた｡,女性,157cm,70Kg,81歳,-,-,141-80mmHg,37.3℃,68bpm,N,N,N,N, P, P2,U,偽性アルドステロン症
```