import azure.functions as func
import logging
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from openai import AzureOpenAI

#AzureOpenAI setup
client_azure = AzureOpenAI(
    api_key = os.getenv("AZURE_KEY"),  
    api_version = "2024-02-15-preview", #this one works for gpt-4o
    azure_endpoint = os.getenv("AZURE_GPT4oMINI_ENDPOINT")
    )

def get_completion_from_messages_azure(messages, 
                                 model=os.getenv("LLM_MODEL"),
                                 temperature=0.1, 
                                 max_tokens=4096):
    response = client_azure.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# Initialize embedings
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    api_key=os.getenv("AZURE_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDINGS_ENDPOINT")
)

#Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "epambot"
namespace = "epambot"
vectorstore = PineconeVectorStore(index_name=index_name,
                                  embedding=embeddings,
                                  namespace=namespace)  


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="get_answer")
def http_trigger_epambot(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    question = req.params.get('question')
    query = f"{question}."

    if not (question):
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            question = req_body.get('question')
            query = f"{question}."

    result = vectorstore.similarity_search(
        query,  # our search query  
        k=10  # return 3 most relevant docs
        )
    context = ""
    for o in result:
        context = o.page_content + context
    
    #Prompt preparation
    delimiter = "####"
    prompt = f"""Act as a person who relocated to The Netherlands and seeking for the answers. Give the answer to the question within provided context: {question}. If you do not have an answer, then just say I do not know."
                """
    messages =  [
        {'role':'system', 
        'content': context},    
        {'role':'user', 
        'content': f"{delimiter}{prompt}{delimiter}"}
        ] 

    answer = get_completion_from_messages_azure(messages)

    if (question):
        return func.HttpResponse(f"{answer}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a question in the query string or in the request body for a personalized response.",
             status_code=200
        )