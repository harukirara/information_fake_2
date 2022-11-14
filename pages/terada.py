import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import random
from janome.tokenizer import Tokenizer

#文章生成での待機時間の表示
@st.cache
def progress_cache(i):
    time.sleep(0.05)

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

#テキストの文を要約し文章生成
def choice(filepath):
    #テキストの要約
    with open(filepath,encoding="utf-8") as file:
        lines = file.readlines()
        s_lines = [line.strip() for line in lines]
    return (random.choice(s_lines))

#タイトルファイルの読み込み関数
@st.cache
def get_list_from_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        return f.read().split()

#タイトルファイルの読み込み
titles = get_list_from_file('./all_data/out_000042/book-titles42.txt')

#読み込み時間を退屈させないために
with st.spinner('Wait for it...'):
    time.sleep(2)
st.snow()

#csvファイルの読み込み
df2=pd.read_csv("./all_data/out_000042/tfidf042.csv")

#類似度を格納する辞書
data_dict={}

#tf-idfやカウントモデルの読み込み
with open("./all_data/out_000042/tfidf042model.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

with open("./all_data/out_000042/count042model.pkl", 'rb') as f:
    counter = pickle.load(f)

#メインの検索部分
st.title("寺田寅彦")
with st.form(key='profile form'):

    #テキストボックス
    query=st.text_input("クエリ")
    
    #形態素解析
    tokenizer=Tokenizer(wakati=True)
    query=tokenizer.tokenize(query)
    query=' '.join(query)
    
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
    dic2 = sorted(data_dict.items(), key=lambda x:x[1],reverse=True)[0:10]


    if submit_btn:
        for x in range(len(dic2)):
            st.markdown('作品名：%s ' % titles[dic2[x][0]])
            st.markdown("類似度%.4f" % dic2[x][1])

            #文章生成
            view_bar(progress_cache)
            st.write("あらすじ")
            st.write(choice("./all_data/generate/42/%d.txt" % dic2[x][0]))
            st.markdown("---") #区切り線
