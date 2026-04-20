### Real time Wikipedia fetching RAG Chatbot with Pinecone

## Quick run
1: Change the '.env.example' file to .env
2: Put the Keys in front of the variables.
3: Install poetry (run in .... in the terminal)
4: Run "poetry run langchain serve" in the terminal
5: run the local terminal " http://localhost:8000/


## 🌟 Features

* RAG-Powered Responses: Answers questions using real time Wikipedia content.
* Conversation Memory: Maintains context across multiple questions
* Modern UI: Clean, responsive chat interface similar to Claude
* Real-time Streaming: Typing indicators and smooth message flow
**Free LLM**: Uses Groq's free API for fast inference

## Pinecone-Serverless

Pinecone Serverless provides usage based pricing and support for unlimited scaling and helps address pain points with vectorstore productionization that can be seen in the community. This repo builds a RAG chain that connects to Pinecone Serverless index using LCEL, it uses HuggingFace embeddings (free), Groq LLM (...)(free) ......... and uses LangSmith to monitor the input / outputs.  

![chain](https://github.com/langchain-ai/pinecone-serverless/assets/122662504/454266ba-727c-4ce0-ae56-7d004c0fb5d4)
Diagram was copied from pinecone serverless & RAG repository. 


Question → Pinecone → Get URLs → Fetch Full Wikipedia Pages → LLM → Answer 
### API keys

Ensure these are set:

* PINECONE_API_KEY
* PINECONE_ENVIRONMENT
* PINECONE_INDEX_NAME 
* GROQ_API_KEY

Note: the choice of embedding model may require additional API keys, such as:
* COHERE_API_KEY
* HUGGINGFACE_API_KEY

### Notebook

For prototyping:
```
poetry run jupyter notebook
```

### Deployment

This repo was created by following these steps:

**(1) Create a LangChain app.**

Run:
```
langchain app new .  
```

This creates two folders:
```
app: This is where LangServe code will live
packages: This is where your chains or agents will live
```

It also creates:
```
Dockerfile: App configurations
pyproject.toml: Project configurations
```

Add your app dependencies to `pyproject.toml` and `poetry.lock` to support Pinecone serverless:
```
poetry add pinecone-client==3.0.0.dev8
poetry add langchain-community==0.0.12
poetry add cohere
poetry add openai
poetry add jupyter
```

Update enviorment based on the updated lock file:
```
poetry install
```

**(2) Add your runnable (RAG app)**

Create a file, `chain.py` with a runnable named `chain` that you want to execute. 

This is our RAG logic (e.g., that we prototyped in our notebook).

Add `chain.py` to `app` directory.

Import the LCEL object in `server.py`:
```
from app.chain import chain as pinecone_wiki_chain
add_routes(app, pinecone_wiki_chain, path="/pinecone-wikipedia")
```

Run locally
```
poetry run langchain serve
```

**(3) Deploy it with hosted LangServe**

Go to your LangSmith console.

Select `New Deployment`.

Specify this Github url.

Add the abovementioned API keys as secrets.
