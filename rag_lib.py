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
            prompt = self.generate_dynamic_prompt(query, combined_docs_content)
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

    def generate_dynamic_prompt(self, query, docs_content):
        return (
            "You are a BenQ product customer service AI. Respond conversationally and accurately about BenQ products in the language of the query. Always use 'we', 'our', and 'us' when referring to BenQ.\n\n"
            f"Customer query: {query}\n\n"
            "Relevant information:\n"
            f"{docs_content}\n\n"
            "Instructions:\n"
            "1. Focus on the exact product model mentioned. Double-check all specifications.\n"
            "2. If information is insufficient or uncertain, state: 'I apologize, but I don't have complete information about that. Please check our official BenQ website for the most up-to-date details.'\n"
            "3. Discuss only BenQ products unless specifically asked about others.\n"
            "4. Adjust your tone to match the customer's style and needs.\n"
            "5. If the query is unrelated to BenQ products or inappropriate, politely redirect the conversation.\n"
            "6. At the end, encourage further questions: 'Do you have any other questions about our BenQ products?'\n\n"
            "Provide a concise, accurate, and helpful response based on these instructions and the available information."
        )
