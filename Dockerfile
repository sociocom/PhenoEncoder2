# pythonはイメージから
FROM python:3.6.9


# mecabの導入
RUN apt-get -y update && \
  apt-get -y upgrade && \
  apt-get install -y mecab && \
  apt-get install -y libmecab-dev && \
  apt-get install -y mecab-ipadic-utf8 && \
  apt-get install -y git && \
  apt-get install -y make && \
  apt-get install -y curl && \
  apt-get install -y xz-utils && \
  apt-get install -y file && \
  apt-get install -y sudo

# mecab-ipadic-NEologdのインストール
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
  cd mecab-ipadic-neologd && \
  ./bin/install-mecab-ipadic-neologd -n -y && \
  echo dicdir = `mecab-config --dicdir`"/mecab-ipadic-neologd">/etc/mecabrc && \
  sudo cp /etc/mecabrc /usr/local/etc && \
  cd ..
EXPOSE 8000

WORKDIR /home
COPY requirement.txt /home
RUN pip install --upgrade pip && \
  pip install -r /home/requirement.txt

COPY ./webapp /home/webapp
WORKDIR /home/webapp

RUN groupadd web
RUN useradd -d /home/webapp -m python
USER python
ENTRYPOINT ["python", "cgiserver.py"]