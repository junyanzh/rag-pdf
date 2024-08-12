import json
import os
from google.cloud import bigquery
from anthropic import AnthropicVertex
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorSearch

class BenQService:
    def __init__(self, status_file='service_status.json', max_queries=300):
        LOCATION = "us-central1"
        self.client = AnthropicVertex(region=LOCATION, project_id="poc-55-genai")
        self.embedding_model = VertexAIEmbeddings(
            model_name="textembedding-gecko@latest",
            project='poc-55-genai'
        )
        self.bq_vector_benq_manual = BigQueryVectorSearch(
            project_id='poc-55-genai',
            dataset_name='benq_poc',
            table_name='benq_manual_embeddings',
            location='US',
            embedding=self.embedding_model,
        )
        self.status_file = status_file
        self.max_queries = max_queries
        self.query_count = self.load_status()

    def load_status(self):
        if not os.path.exists(self.status_file):
            self.save_status(0)
        with open(self.status_file, 'r') as f:
            return json.load(f)['query_count']

    def save_status(self, count):
        with open(self.status_file, 'w') as f:
            json.dump({'query_count': count}, f)

    def search_and_generate_answer(self, query):
        if self.query_count >= self.max_queries:
            return "You have reached the maximum number of questions. Please contact system administrators, aka Willy & Colin @55."
        
        query_vector = self.embedding_model.embed_query(query)
        docs = self.bq_vector_benq_manual.similarity_search_by_vector(query_vector, k=3)
        relevant_docs = [doc.page_content for doc in docs]
        if relevant_docs:
            combined_docs_content = "\n\n".join(relevant_docs)
            prompt = (
                f"You are a professional BenQ product customer service representative, assisting customers via a chat interface. Your role is to provide helpful, accurate responses in a conversational tone, avoiding formal or email-like responses.\n\n"
                f"Firstly, address the customer's question '{query}'. If the customer is seeking information that inherently involves comparison (such as differences between models or product series), guide them to use our comparison tool on the official website at https://www.benq.com/, where they can compare features and specifications of different BenQ products side-by-side.\n\n"
                f"Additionally, check if the information from our documents aligns with the customer's question. If the query is broad or lacks details, kindly ask the customer for more specific information or a clearer context of their issue. Present the information retrieved:\n\n"
                f"{combined_docs_content}\n\n"
                f"If this information does not sufficiently answer the question, or if the models mentioned do not match, respond kindly and inform the customer that we do not have enough information to fully resolve their query. Suggest visiting our official site for more detailed information.\n\n"
                f"Always focus responses on BenQ products, avoiding references to other brands. If the customer expresses a strong intention to purchase a BenQ product or accessory, encourage visiting our official site. Otherwise, address their inquiries based on the available information."
            )
            message = self.client.messages.create(
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                model="claude-3-haiku@20240307"
            )
            response_text = message.content[0].text
            self.query_count += 1
            self.save_status(self.query_count)
            return response_text
        else:
            return "Sorry, no relevant information could be found for your query."
