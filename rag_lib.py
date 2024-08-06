from google.cloud import bigquery
import os
from anthropic import AnthropicVertex
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorSearch

# 初始化AnthropicVertex客户端
LOCATION = "us-central1"  # 或 "europe-west4"
client = AnthropicVertex(region=LOCATION, project_id="poc-55-genai")

# 初始化嵌入模型和 BigQuery 向量搜索
embedding_model = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest",
    project='poc-55-genai'
)

bq_vector_benq_manual = BigQueryVectorSearch(
    project_id='poc-55-genai',
    dataset_name='benq_poc',
    table_name='benq_manual_embeddings',
    location='US',
    embedding=embedding_model,
)

def search_and_generate_answer(query):
    query_vector = embedding_model.embed_query(query)
    docs = bq_vector_benq_manual.similarity_search_by_vector(query_vector, k=3)
    relevant_docs = [doc.page_content for doc in docs]
    
    if relevant_docs:
        combined_docs_content = "\n\n".join(relevant_docs)
        prompt = (
            f"You are a professional BenQ product customer service representative. Now you will reply to customer in a chatbot. Please don't use official mail like reponse, answer like helper "
            
            f"You need to respond to the customer's question '{query}'. Make sure the question is talking about BenQ products, if no, please ask customers to provide product name or series number kindly.\n\n "
            f"Below is the information retrieved from the vector database:\n\n:"
            f"{combined_docs_content}\n\n"
            f"Please respond kindly, and if the docments provided from vector databade are un-related to user's question, especialy the model are not matched, please inform the user accordingly, said like: Sorry, no relevant information could be found."

            f"if the customer's query show strong intention to buy BenQ product(include accessories), then direct them to officail site ( https://www.benq.com/). Else, just answer the questions."
        )
        
        # 使用 Claude API 生成回答
        message = client.messages.create(
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="claude-3-haiku@20240307",  # 根据需要选择合适的模型版本
        )
        
        # 获取回答中的文字部分
        response_text = message.content[0].text
        return response_text
    else:
        return "Sorry, no relevant information could be found for your query."
