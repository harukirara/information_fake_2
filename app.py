import streamlit as st
from PIL import Image

st.title("文書検索アプリ")
st.caption("あらすじを信じるのはあなた次第")

num=3
lcol=[]
col=st.columns(num)


akutagawa=Image.open("./data/ch04/image/akutagawa_ryunosuke.png")
masaoka=Image.open("./data/ch04/image/masaoka_shiki.png")
natsume=Image.open("./data/ch04/image/natsume_souseki.png")
pnglist=[akutagawa,masaoka,natsume]
for i,k in enumerate(pnglist):
    with col[i]:
        st.image(k,width=200)