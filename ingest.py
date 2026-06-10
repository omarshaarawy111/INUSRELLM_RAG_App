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
 
# Load Open AI key
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configuraions
DB_NAME = "preprocessed_db"
collection_name = "docs"
# The embedding model of Open AI
embedding_model = "text-embedding-3-large"
# Path of knowledge base to create object represents the path
KNOWLEDGE_BASE_PATH = Path("knowledge-base")
# The size of chunk per character
AVERAGE_CHUNK_SIZE = 500
# The generation model of Open AI
GENERATOR_MODEL = "gpt-4.1-nano"
# Open AI instance
openai = OpenAI(api_key=openai_api_key)
# wait_exponential: here we specify the waiting time before retry and another
# multiplier: multi * 2 ^ retry_time => retry_time starts at 0 and increase by 1 with each retry, min: min wait 10 seconds, max: min wait 240 seconds
wait = wait_exponential(multiplier=1, min=10, max=240)
# Number of CPU processors
WORKERS = 3

# Classes
# Create something inspired by Lang Chain
# In Lang Chain we have document class which has page content and metadata but here we create our own class with the same properties from scratch
# So we return this Result class object instead of Lang Chain document class object
class Result(BaseModel):
    # page_content is  string while metadata is dictionary to store any meta data we want to add
    page_content: str
    metadata: dict

# Create validation class from scratch so we could see BaseModel got inherited
# This is the structured output we want from chunking step and the validation as well
class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    # Function to convert chunk object to result object 
    # We pass the orginal document which chunk comes from to add meta data manually
    def as_result(self, document):
        # Add meta data manually
        # In Lang Chain metadata of source is added automatically but here we have to add it manually
        # In both Lang Chain and here type is added manually for better filtering
        metadata = {"source": document["source"], "type": document["type"]}
        # Page content is the combination of attributes
        return Result(page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,metadata=metadata)

# Create class to store list of chunks
class Chunks(BaseModel):
    chunks: list[Chunk]

# This is similar to Lang Chain DirectoryLoader
def fetch_documents():
    documents = []
    # Loop over directory
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        # The same as os.path.basename(folder) to get the name of the subfolder which is 1 of 4 folders to be added to meta data
        doc_type = folder.name
        # Loop over .md files
        for file in folder.rglob("*.md"):
            # Open each file
            with open(file, "r", encoding="utf-8") as f:
                # In Lang Chain through loader.load() we could fetch text and metadata together but here we have to do it manually
                # file.as_posix(): converts a Python pathlib Path object into a string representation with standard forward slashes (/) as the directory instead of forward slashes (\) in windows 
                # then we have list of dictionaries as each dictionary represents one document with its type, source and text to be passed to chunking and embedding steps
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents

# Chunking step 
# Instead of letting Lang Chain do the chunking for us we create our own chunking function 
# First create prompt that will be passed to LLM to do the chunking for us
def make_prompt(document):
    # The number of chunks will be the total number of characters in the document divided by the average chunk size we want to have 
    # We add 1 to make sure about covering all content as we work with integer division
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""

# Create user prompt to be passed to LLM to do the chunking for us
def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]

# Don't forget to add wait decorator
# Apply chunking step for each document
@retry(wait=wait)
def process_document(document):
    # Create user prompt by creating prompt first
    messages = make_messages(document)
    # Pass user prompt to LLM and get the response which is the chunks in json format
    # complilation is like Open AI chat completion but it's more general for Lite LLM
    # We have two phases
    # first is structured output generation so we pass the Chunks class to get this kind of reposnse in json
    response = completion(model=GENERATOR_MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    # After forcing the model to return the structured output in json format 
    # Second phase is validation 
    # model_validate_json() to validate the format of the json response and make sure about the format of data 
    # Chunks: class that handle list of chunks 
    # chunks: attribute of Chunks class to handle list of chunk objects
    # We need the attribute not the class itself because we want to work with list of chunk objects 
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    # We convert each chunk object to result object and return list of result objects which are simliar to Lang Chain objects as the final output of chunking step to be passed to embedding 
    # Now we use chunk class as we work per chunk 
    return [chunk.as_result(document) for chunk in doc_as_chunks]

# Accumelate all chunks from all documents in one list to be passed to embedding step
def create_chunks(documents):
    chunks = []
    # tqdm to show progress bar at each iteration of loop
    with Pool(processes=WORKERS) as pool:
        # Here imap_unordered(process_document, documents) means apply process_document on each element of docuemtns in parallel
        # Why we work with imap_unorderd: we have two types (map) which keeping ordering of documents and another one is (imap_unordered) which is the opposite
        # So when finishing one document it returns it instantly and this is faster than ordering
        # total: total number for tqdm bar
        for doc in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            # Becasue we return list of chunks per document so to add lists to list we use extend
            chunks.extend(doc)
    return chunks

# Create embedding then create vector database and store data inside it without Lang Chain
def create_embeddings(chunks):
    # Create vector database with Chroma library directly without Lang Chain
    chroma = PersistentClient(path=DB_NAME)
    # The same step of checking if Exisitng database is there then delete it as we don't know if there is change in data or embedding model
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
        
    # Prepare data for storing in vector database
    # Create list of indexes
    ids = [str(i) for i in range(len(chunks))]
    # We need to extract text from chunks 
    texts = [chunk.page_content for chunk in chunks]
    # embeddings step with OpenAI embedding model to get vectors for all chunks
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    # Now we have vectors in the same order of chunks 
    # We need vectors
    vectors = [e.embedding for e in emb]
    # We need to extract meta data from chunks
    metas = [chunk.metadata for chunk in chunks]  
    
    # Create vector database
    collection = chroma.get_or_create_collection(collection_name)
    # Store data in vector database
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {len(chunks)} chunks, {collection.count()} embeddings, and {len(vectors[0])} dimensions")


if __name__ == "__main__":
    # Ingestion Phase
    # Steps of loading documents from knowledge base ---> chunking ---> Chunk validation class ---> Result class ---> embedding and vector database creation
    # Loading documents
    documents = fetch_documents()
    # Chunking
    chunks = create_chunks(documents)
    # Embedding and vector database creation
    create_embeddings(chunks)
    print("Ingestion complete")