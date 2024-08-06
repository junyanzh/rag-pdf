import streamlit as st
import benq_rag_lib as glib  # Reference to your modified library script

st.set_page_config(page_title="BenQ Customer Service Bot")
st.title("BenQ Customer Service Bot")

# 設定問題次數上限
MAX_QUERIES = 2  # 你可以根據需要更改這個數字

# 初始化問題次數，如果不存在則設為0
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# 更新前端顯示，已試次數和上限
attempts_display = f"Attempted Questions: {st.session_state.query_count}/{MAX_QUERIES}"
st.write(attempts_display)

# 檢查是否已達提問次數上限
if st.session_state.query_count < MAX_QUERIES:
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Fetching your answer..."):
            answer = glib.search_and_generate_answer(query)
            st.write(answer)
        # 問題計數器增加
        st.session_state.query_count += 1
else:
    st.error("You have reached the maximum number of questions. Please contact the administrator to reset the count.")
