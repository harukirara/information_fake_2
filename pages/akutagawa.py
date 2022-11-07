import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import random
import setting
from lxml.html._diffcommand import read_file
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import T5Tokenizer, AutoModelForCausalLM

#文章生成での待機時間の表示
def progress_cache(i):
    time.sleep(0.025)

#プログレスバーの表示
def view_bar(func):
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        func(i)

#テキストの続きを生成
def make_next(prompt):
    answer=""
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    outputs = model.generate(
        inputs, 
        max_length=200, 
        do_sample=True, 
        top_p=0.95, 
        top_k=60, 
        no_repeat_ngram_size=2,
        num_beams=5, 
        early_stopping=True,
        num_return_sequences=1
    )
    for i, beam_output in enumerate(outputs):
        answer=answer+tokenizer.decode(beam_output, skip_special_tokens=True)
    return answer

#テキストの文を要約し文章生成
@st.cache
def choice(filepath):
    #テキストの要約
    text = read_file(filepath)
    parser = PlaintextParser.from_string(text, Tokenizer('japanese'))
    summarizer = LexRankSummarizer()
    res = summarizer(document=parser.document, sentences_count=2)
    plut=""
    for sentence in res:
        plut=plut+str(sentence)
    if plut=="":
        plut="今日"
    #transformerで補足
    return make_next(plut)

#タイトルファイルの読み込み関数
@st.cache
def get_list_from_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        return f.read().split()

#タイトルファイルの読み込み
titles = get_list_from_file('./data/ch04/out_000879/book-titles879.txt')

#読み込み時間を退屈させないために
with st.spinner('Wait for it...'):
    time.sleep(2)
st.balloons()
st.snow()

#csvファイルの読み込み
df2=pd.read_csv("./data/ch04/out_000879/tfidf879.csv")

#類似度を格納する辞書
data_dict={}

#tf-idfやカウントモデルの読み込み
with open("./data/ch04/out_000879/tfidf879model.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

with open("./data/ch04/out_000879/count879model.pkl", 'rb') as f:
    counter = pickle.load(f)

#メインの検索部分
st.title("芥川龍之介")
with st.form(key='profile form'):

    #テキストボックス
    query=st.text_input("クエリ")
    #ベクトルに変換
    mozinum=np.array([query])
    instr=counter.transform(mozinum)
    x= vectorizer.transform(instr)

    #ボタン
    submit_btn=st.form_submit_button("検索")
    cancel_btn=st.form_submit_button("リセット")

    #すべてのデータに関してコサイン類似度の計算
    for index, row in df2.iterrows():
        data=np.array(row[1:])
        new_data=data.reshape((1,-1))
        cs = cosine_similarity(new_data,x)  # cos類似度計算
        data_dict[index]=cs

    #類似度の降順にソート,表示させるのは10件
    dic2 = sorted(data_dict.items(), key=lambda x:x[1],reverse=True)[0:5]


    if submit_btn:
        for x in range(len(dic2)):
            st.markdown('作品名：%s ' % titles[dic2[x][0]])
            st.markdown("類似度%.4f" % dic2[x][1])

            #文章生成
            view_bar(progress_cache)
            st.write("あらすじ")
            st.write(choice("./data/ch04/out_000879/%d.txt" % dic2[x][0]))
            st.markdown("---") #区切り線