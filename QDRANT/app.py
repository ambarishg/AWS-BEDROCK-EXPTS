import streamlit as st
from rag_qdrant_bedrock import RAG_QDRANT_BEDROCK

st.header('Search Engine with Qdrant and Bedrock')

user_input = st.text_input('Enter your question here:', 
                           'What is Diploblastic and Triploblastic Organisation ?')

if st.button('Submit'):
    rqb = RAG_QDRANT_BEDROCK()
    st.write(rqb.query(user_input))
    