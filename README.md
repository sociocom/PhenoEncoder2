# PhenoEncoder2
## 概要
脳腫瘍の原因となる糖尿病と喫煙情報に関係する病名抽出と患者の病名抽出をするツールです．\
csvファイルからカルテ要約と病名を抽出します．

## 手法
ルールベースにより患者の要約を作成し，SVMを使用して病名の抽出します。

詳細は[こちら](https://www.jstage.jst.go.jp/article/jsaisigtwo/2018/AIMED-006/2018_01/_article/-char/ja/)の論文をご覧ください


## 環境
```
Python 3.6.9
```
その他パッケージは `requirement.txt` をご覧ください．

## ローカルでの実行方法
Type following command on terminal.
```
git clone dgit@github.com:sociocom/PhenoEncoder2.git
cd PhenoEncoder2
python cgiserver.py
```
Then, access to http://localhost:8000 .

### 環境構築メモ
***PhenoEncoder2***を構築するためのメモです．

筆者は *macOS* に *pyenv* と *venv* を用いて構築しました．
`PE_2.py` で *import* されているパッケージを順に `pip install hogehoge` をしていけば，基本同じパッケージがインストールできます．
※一部ダウングレードしたパッケージをしてしないといけない場合があったので，注意点を参照すること．

#### 注意点
pythonのバージョンは **3.6.9** で構築しています．
いくつかのパッケージは随時 `git clone hogehoge` したのち，`setup.py` を実行した記憶があります．

筆者がインストールしたパッケージは同階層にある `requirement.txt` に記録しています．
もし構築時にエラーが出た際は `requirement.txt` を参照して，バージョンと相違ないか確認してください．

同階層の `cgiserver.py` を実行したのち，http://localhost:8000 にアクセスするとデモ操作ができます．

### 動作の確認について
index.html内のtextareaに改行区切りで入力された複数ドキュメントに対して要約，病名抽出します．
単一ドキュメントではありますが，動作チェックを行いたい場合は[デモ環境](http://aoi.naist.jp/~shibata/PhenoEncoder/sample%20app/)にアクセスすると確認ができます．

#### 注意点
階層が違うところで `cgiserver.py` を実行すると，実行した階層の `index.html` を開こうとするので注意すること．