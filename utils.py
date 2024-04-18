from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import OpenAI
import os
import langchain
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import warnings
warnings.filterwarnings('ignore')

#os.environ["OPENAI_API_KEY"] = ""
langchain.debug = True

print('Loading and chunking documents....')
loaders = [TextLoader('data/Apollo_mission_success.txt')]

docs = []
for l in loaders:
    docs.extend(l.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(docs)
print(f'Docs chunked successfully! Total docs are {len(texts)}')

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}
hyde_llm = OpenAI()

print('Loading Embeddings model....')
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    # model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)
print('Embeddings model loaded!')

hyde = HypotheticalDocumentEmbedder.from_llm(hyde_llm, bge_embeddings, prompt_key="web_search")

hyde.llm_chain.prompt.template = '''Generate the most plausible response to the question to the best of your ability.
Question: {QUESTION}
Response: '''

vectorstore = Chroma.from_documents(documents=texts, embedding=hyde)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    text = "\n\n".join(doc.page_content for doc in docs)
    return text

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_hyde_response(query):
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        print('Error in getting the response: ', e)
        return 'Sorry, Please try again after sometime!'
    

normal_query_vectorstore = Chroma.from_documents(documents=texts, embedding=bge_embeddings)
normal_retriever = normal_query_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
normal_query_rag_chain = (
    {"context": normal_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_normal_retriever_response(query):
    try:
        response = normal_query_rag_chain.invoke(query)
        return response
    except Exception as e:
        print('Error in getting the response: ', e)
        return 'Sorry, Please try again after sometime!'