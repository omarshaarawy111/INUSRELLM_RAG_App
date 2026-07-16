# Work with no Lang Chain
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
# Use pydantic to create classes and apply validation to make sure about the format of data we return from retrival (schema) and pass it to LLM
from pydantic import BaseModel, Field
# This is the chroma database library directyl not through Langchain
from chromadb import PersistentClient
from tqdm import tqdm
# For LLM abstraction
# Remeber Lite LLM is just another framework has mulitple abstraction layers
from litellm import completion
# This is for multiprocessing so working on vairous chunks at the same time (parallel) instead of concurrent
# We could make things faster 
# We create processes and each one is independent python interpreter working with specific CPU core (couldn't be guaranteed as OS schedular decides)
from multiprocessing import Pool
# Python decorators are functions that add behaviour to another function without changing it's internal code
# It could be written in two ways: (func = decorator(func)) and (@decorator before func) 
# tenacity is very convenient library that is very helpful to add python decorator @retry
# @retry important if chunking function (LLM model function) could throguh RateLimitError so it retrys slowly and slowly (exponential backoff)
# Exponential backoff means we wait 1 2 4 8 16 etc of seconds
# RateLimitError means too many APIs requests
from tenacity import retry, wait_exponential

# Steps: user query ---> query rewriting ---> query expansion ---> query embedding ---> retriever ---> top-k chunks ---> merge chunks ---> reranking ---> top-s chunks ---> RankOrder validation class ---> prompt ---> LLM ---> answer

# Load Open AI key
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configuraions
DB_NAME = "preprocessed_db"
collection_name = "docs"
# The embedding model of Open AI
embedding_model = "text-embedding-3-large"
# The generation model of Open AI
GENERATOR_MODEL = "gpt-4.1-nano"
# Open AI instance
openai = OpenAI(api_key=openai_api_key)

# Initialize vector database
chroma_client = PersistentClient(path=DB_NAME)
collection = chroma_client.get_or_create_collection(name=collection_name)
# wait_exponential: here we specify the waiting time before retry and another
# multiplier: multi * 2 ^ retry_time => retry_time starts at 0 and increase by 1 with each retry, min: min wait 10 seconds, max: min wait 240 seconds
wait = wait_exponential(multiplier=1, min=10, max=240)
# Number of CPU processors
WORKERS = 3
RETRIEVAL_K = 10
# Trim the results to number of specific number of s after reranking
FINAL_K = 10
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}
With this context, please answer the user's question. Be accurate, relevant and complete.
"""


# Classes 
# Create something inspired by Lang Chain
# In Lang Chain we have document class which has page content and metadata but here we create our own class with the same properties from scratch
# So we return this Result class object instead of Lang Chain document class object
class Result(BaseModel):
    # page_content is  string while metadata is dictionary to store any meta data we want to add
    page_content: str
    metadata: dict

# Rearanking 
# Create validation class from scratch so we could see BaseModel got inherited
# This is the structured output we want from chunking step and the validation as well
class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

# Rewriting query step which is before query embedding
@retry(wait=wait)
def rewrite_query(question, history=[]):
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=GENERATOR_MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

# These is equivelant to retriever.invoke() 
@retry(wait=wait)
def fetch_context_unranked(question):
    # Query embedding
    # To access embeddigs we need to access key of data with its first element from the whole response
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    # We dive into vector database collection to get the relevant chunks based on query embedding 
    # Retriever to get top-k 
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    # Unranked chunks
    chunks = []
    # results return two lists one for documents and one for meta data and we need to loop over them together to get the text and meta data of each chunk 
    # [0] because we pass one query only so we have one list of results for this query
    for result in zip(results["documents"][0], results["metadatas"][0]):
        # As awlways for the second time we need the Result class to have the same format of data in the step of emeddings
        # So we have chunks as Result object wich is closer to Lang Chain object 
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


# Merge queries for reranking then
# (chunks1, chunks2)
# Return the unique chunks which is all chunks 1 of orginal question and the difference in chunks of rewritten questions
# We suspect the error may issued by rewritten chunks
def merge_chunks(chunks1, chunks2):
    merged = chunks1[:]
    existing = [chunk.page_content for chunk in chunks1]
    for chunk in chunks2:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

# Rerank using LLM after retrival to improve the order of chunks based on relevance to the question
# Question is the reference for reranking
@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    # User prompt is the question + the list of chunks with their ids to be ranked + additional contex
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    # Retrieve chunks and ids
    for index, chunk in enumerate(chunks):
        # We need to get text from each chunk to chove it with prompt
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Pass user prompt to LLM and get the response which is the reranked chunks in json format
    # complilation is like Open AI chat completion but it's more general for Lite LLM
    # We have two phases
    # first is structured output generation so we pass the RankOrder class to get this kind of reposnse in json
    response = completion(model=GENERATOR_MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    # After forcing the model to return the structured output in json format 
    # Second phase is validation 
    # model_validate_json() to validate the format of the json response and make sure about the format of data 
    # RankOrder: class that handle the ranking order
    # order: attribute of RankOrder class to handle the list of chunk ids
    # We need the attribute not the class itself because we want to work with the list of chunk ids
    order = RankOrder.model_validate_json(reply).order
    # Get the relavant chunk based on id in order
    return [chunks[i - 1] for i in order]

# Steps: query ---> query rewriting ---> query expansion ---> query embedding ---> retriever ---> top-k chunks ---> merge chunks ---> reranking ---> top-s chunks ---> RankOrder validation class
def fetch_context(original_question, rewritten_question):
    # We use both orginal and rewritten queries for expansion then reranking
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    # Merge queries
    chunks = merge_chunks(chunks1, chunks2)
    # Rerank
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]

# Create prompt
def make_rag_messages(question, history, chunks):
    # Context is source + content of each chunk
    context = "\n\n".join(f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks)
    # Shove addtional context to system prompt
    system_prompt = SYSTEM_PROMPT.format(context=context)
    # Question is user prompt
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

# General function for all steps of answer: user query ---> query rewriting ---> query expansion ---> query embedding ---> retriever ---> top-k chunks ---> merge chunks ---> reranking ---> top-s chunks ---> RankOrder validation class ---> prompt ---> LLM ---> answer
# () -> return value
@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    # Query rewriting
    query = rewrite_query(question, history)
    # Print step
    print(query)
    # Query embedding ---> retriever ---> top-k chunks ---> reranking ---> top-s chunks ---> RankOrder validation class
    # Print step
    chunks = fetch_context(question, query)
    # prompt ---> LLM ---> answer
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=GENERATOR_MODEL, messages=messages)
    # Return string which is response, list of chunks that we retrieved answer from
    return response.choices[0].message.content, chunks