# This application is a great example of integrating several powerful technologies:
#
#
 # The LLM (VLLMOpenAI): This is the creative engine. It's the Mistral model running on a VLLM server, 
 # responsible for generating human-like text.
 # The Vector Database (Milvus): This is the long-term memory. 
 # It stores numerical representations (embeddings) of your documents, 
 # allowing for incredibly fast similarity searches.
 # The Translator (HuggingFaceEmbeddings): This model's job is to turn human text into the vectors that Milvus understands.
 # The Orchestrator (LangChain): This is the glue that holds everything together. 
 # It provides the building blocks like chains and retrievers that make it easy 
 # to connect all these different components into a single, coherent application.

import json
import os
import random
import time
import httpx
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional, List, Dict, Any


import gradio as gr
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLMOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from milvus_retriever_with_score_threshold import MilvusRetrieverWithScoreThreshold

load_dotenv()

# Parameters

APP_TITLE = os.getenv('APP_TITLE', 'Chat with your Knowledge Base!')
SHOW_TITLE_IMAGE = 'True' #os.getenv('SHOW_TITLE_IMAGE', 'True')

INFERENCE_SERVER_URL = 'https://URL_OF_THE_LLM' #os.getenv('INFERENCE_SERVER_URL')
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2' #os.getenv('MODEL_NAME')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 512))
TOP_P = float(os.getenv('TOP_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
PRESENCE_PENALTY = float(os.getenv('PRESENCE_PENALTY', 1.03))

MILVUS_HOST = 'URL_OF_MILVUS' #os.getenv('MILVUS_HOST')
MILVUS_PORT = 19530 #os.getenv('MILVUS_PORT')
MILVUS_USERNAME = 'root' #os.getenv('MILVUS_USERNAME')
MILVUS_PASSWORD = 'Milvus' #os.getenv('MILVUS_PASSWORD')
MILVUS_COLLECTIONS_FILE = './colletion_files.txt' #os.getenv('MILVUS_COLLECTIONS_FILE')
NO_KB_OPTION = "NONE"

DEFAULT_COLLECTION = os.getenv('DEFAULT_COLLECTION')
PROMPT_FILE = os.getenv('PROMPT_FILE', 'prompt.txt')
MAX_RETRIEVED_DOCS = int(os.getenv('MAX_RETRIEVED_DOCS', 4))
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.99))


# Load collections from JSON file
with open(MILVUS_COLLECTIONS_FILE, 'r') as file:
    collections_data = json.load(file)

# Load Prompt template from txt file
with open(PROMPT_FILE, 'r') as file:
    prompt_template = file.read()

############################
# Streaming call functions #
############################
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list


    # A Queue is needed for Streaming implementation
    q = Queue()

    # Instantiate LLM
    llm =  VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=INFERENCE_SERVER_URL,
        model_name=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        streaming=True,
        verbose=False,
        callbacks=[QueueCallback(q)],
        async_client=httpx.AsyncClient(verify=False),
        http_client=httpx.Client(verify=False)
    )
    # 1. First, create the Milvus vector store instance
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=selected_collection,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
        consistency_level="Session",
        metadata_field="metadata",
        text_field="page_content"
    )

    # 2. Then, create the retriever from the vector store, specifying the search type and score threshold
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'k': MAX_RETRIEVED_DOCS,
            'score_threshold': SCORE_THRESHOLD
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        #retriever=retriever,
        chain_type_kwargs={"prompt": qa_chain_prompt},
        return_source_documents=True
        )

    # Create a Queue
    job_done = object()


    # Create a function to call - this will run in a thread

def task():
    # Use the .stream() method and loop through the response chunks
    full_response = ""
    source_documents = []
    for chunk in qa_chain.stream({"query": input_text}):
        # The stream method yields dictionaries. We check what's in them.
        if "result" in chunk:
            # This is a token from the LLM's answer
            token = chunk["result"]
            q.put(token)
            full_response += token
        if "source_documents" in chunk:
            # The source documents are usually passed along with the chunks
            source_documents = chunk["source_documents"]

    # Now that the stream is finished, process the sources
    if source_documents:
        sources = remove_source_duplicates(source_documents)
        if len(sources) != 0:
            q.put("\n\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
    
    # Signal that the job is done
    q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue




def stream(input_text, selected_collection) -> Generator:
    # A Queue is needed for Streaming implementation
    q = Queue()

    # Instantiate LLM (this is common for both cases)
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=INFERENCE_SERVER_URL,
        model_name=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        streaming=True,
        verbose=False,
        callbacks=[QueueCallback(q)],
        async_client=httpx.AsyncClient(verify=False),
        http_client=httpx.Client(verify=False)
    )

    job_done = object()

    # --- START: CONDITIONAL LOGIC ---

    if selected_collection == NO_KB_OPTION:
        # --- CASE 1: NO KNOWLEDGE BASE ---
        # Use a simple prompt for direct chat
        direct_prompt = PromptTemplate.from_template(
            "You are a helpful AI assistant. Answer the following question.\nQuestion: {question}"
        )
        
        # Create a simple chain that just combines the prompt and the LLM
        llm_chain = LLMChain(llm=llm, prompt=direct_prompt)

        def task():
            # Invoke the chain; streaming is handled by the callback
            llm_chain.invoke({"question": input_text})
            q.put(job_done)

    else:
        # --- CASE 2: KNOWLEDGE BASE SELECTED (RAG Logic) ---
        # 1. Create the Milvus vector store instance
        vector_db = Milvus(
            embedding_function=embeddings,
            collection_name=selected_collection,
            connection_args={
                "host": MILVUS_HOST, 
                "port": str(MILVUS_PORT),
                "user": MILVUS_USERNAME, 
                "password": MILVUS_PASSWORD,
                "secure": False
            },
            consistency_level="Session",
            metadata_field="metadata",
            text_field="page_content"
        )

        # 2. Create the retriever with the corrected, simpler search type
        # We simply hire a Librarian The code first establishes a connection to your Milvus vector database. .
        retriever = vector_db.as_retriever(
            search_type="similarity",  # <-- FIX 1: Use the simpler search type
            search_kwargs={'k': MAX_RETRIEVED_DOCS}
        )
        
        # 3. Instantiate the QA chain
        # Build the Assembly Line (RetrievalQA): This is the main chain. 
        # It orchestrates the entire RAG process: it sends the question to the retriever, 
        # gets the relevant documents back, combines those documents with the original question into a new, 
        # more detailed prompt, and finally sends that rich prompt to the LLM.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_chain_prompt},
            return_source_documents=True
        )

        # 4. Define the task using the correct .stream() method
        def task():
            # Use the .stream() method and loop through the response chunks
            source_documents = []
            for chunk in qa_chain.stream({"query": input_text}): # <-- FIX 2: Use .stream()
                # The stream method yields dictionaries. We check what's in them.
                if "result" in chunk:
                    # This is a token from the LLM's answer
                    q.put(chunk["result"])
                if "source_documents" in chunk:
                    # The source documents are usually passed along with the chunks
                    source_documents = chunk["source_documents"]

            # Now that the stream is finished, process the sources
            if source_documents:
                sources = remove_source_duplicates(source_documents)
                if len(sources) != 0:
                    q.put("\n\n*Sources:* \n")
                    for source in sources:
                        q.put("* " + str(source) + "\n")
            
            # Signal that the job is done
            q.put(job_done)

    # --- END: CONDITIONAL LOGIC ---

    # Create a thread and start the function (this part is the same for both cases)
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

######################
# LLM chain elements #
######################

# Document store: Milvus
model_kwargs = {'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs=model_kwargs,
    show_progress=False
)

# Prompt
qa_chain_prompt = PromptTemplate.from_template(prompt_template)


####################
# Gradio interface #
####################

#collection_options = [(collection['display_name'], collection['name']) for collection in collections_data]

# This is the corrected code
collection_options = [("None (Direct to LLM)", NO_KB_OPTION)] + \
                     [(collection['display_name'], collection['name']) for collection in collections_data]

# Set the default to be the "None" option
DEFAULT_COLLECTION = NO_KB_OPTION

def select_collection(collection_name, selected_collection):
    return {
        selected_collection_var: collection_name
        }

def ask_llm(message, history, selected_collection):
    for next_token, content in stream(message, selected_collection):
        yield(content)

css = """
footer {visibility: hidden}
.title_image img {width: 80px !important}
"""

with gr.Blocks(title="Knowledge base backed Chatbot", css=css) as demo:
    selected_collection_var = gr.State(DEFAULT_COLLECTION)
    with gr.Row():
        if SHOW_TITLE_IMAGE == 'True':
            gr.Markdown(f"# ![image](/file=./assets/logo.png)   {APP_TITLE}")
        else:
            gr.Markdown(f"# {APP_TITLE}")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"This chatbot lets you chat with a Large Language Model (LLM) that can be backed by different knowledge bases (or none).")
            collection = gr.Dropdown(
                choices=collection_options,
                label="Knowledge Base:",
                value=DEFAULT_COLLECTION,
                interactive=True,
                info="Choose the knowledge base the LLM will have access to:"
            )
            collection.input(select_collection, inputs=[collection,selected_collection_var], outputs=[selected_collection_var]),
            gr.Markdown(f"Inference server: {INFERENCE_SERVER_URL}")
            gr.Markdown(f"Model: {MODEL_NAME}")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=(None,'assets/logo.png'),
                render=False,
                show_copy_button=True
                )
            gr.ChatInterface(
                ask_llm,
                additional_inputs=[selected_collection_var],
                chatbot=chatbot,
                stop_btn=None,
                description=None
                )

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=10
        ).launch(
        server_name='0.0.0.0',
        share=False,
        #favicon_path='./assets/robot-head.ico',
        allowed_paths=["./assets/"]
        )
