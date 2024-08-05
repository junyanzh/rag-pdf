import streamlit as st
import benq_rag_lib as glib  # Reference to your modified library script

st.set_page_config(page_title="BenQ Customer Service Bot")
st.title("BenQ Customer Service Bot")

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Fetching your answer..."):
        answer = glib.search_and_generate_answer(query)
        st.write(answer)
