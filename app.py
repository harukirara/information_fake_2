import streamlit as st
from PIL import Image

st.title("文書検索アプリ")
st.caption("あらすじを信じるのはあなた次第")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("芥川龍之介")
    st.image("./all_data/image/akutagawa_ryunosuke.png", use_column_width=True,width=200)

with col2:
    st.header("太宰治")
    st.image("./all_data/image/dazai_osamu.png", use_column_width=True,width=200)
    
with col3:
    st.header("正岡子規")
    st.image("./all_data/image/masaoka_shiki.png", use_column_width=True,width=200)

col1, col2, col3 = st.columns(3)

with col1:
    st.header("宮沢賢治")
    st.image("./all_data/image/nigaoe_miyazawa_kenji.png", use_column_width=True,width=200)

with col2:
    st.header("夏目漱石")
    st.image("./all_data/image/natsume_souseki.png", use_column_width=True,width=200)
    
with col3:
    st.header("寺田寅彦")
    st.image("./all_data/image/eto_tora_banzai.png", use_column_width=True,width=200)
