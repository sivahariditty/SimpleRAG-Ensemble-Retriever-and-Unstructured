import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import re
#Check for unstructured package
try:
    from unstructured_client import UnstructuredClient
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
    vec_method = 'UNSTRUCT'
except:
    vec_method = 'LANGC'

from langchain_core.documents import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever


class RAG:

    def __init__(self):
        self._OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self._embeddings = OpenAIEmbeddings()
        self._llm = ChatOpenAI(model="gpt-4o", temperature = 0.75)

    def main(self):
        """
        Execute the full Retrieval-Augmented Generation (RAG) pipeline to process a list of test questions from (test_questions.txt)
        generate answers for each, and extract relevant contexts from the original the pdf text sources (eBook.pdf).

        This method orchestrates the workflow of the RAG pipeline, starting from question processing,
        through context retrieval, answer generation, and finally compiling the results into a structured JSON format.
        The JSON object includes the original questions, their corresponding answers, and the contexts used to generate these answers.

        Note: The results will be assessed using the 'ragas' library, focusing on the following metrics:
            - Faithfulness: The degree to which the generated answers accurately reflect the information in the context.
            - Answer Relevancy: How relevant the generated answers are to the input questions.
            - Context Utilization: The accuracy and relevance of the contexts retrieved to inform the answer generation.

        Returns:
            dict: A JSON-structured dictionary containing the following keys:
                - "question": A list of the input test questions.(the 3 questions from the test_questions.txt)
                - "answer": A list of generated answers corresponding to each question.
                - "contexts": The extracted contexts from original text sources that were used to inform the generation of each answer.

        """
        # Implementation of the RAG pipeline goes here
        return_dict = {}
        questions = []
        answers = []
        contexts = []
        sts = self.vectorise_doc_unstruct('app/eBook.pdf') ## STEP 1 calls document splitting and vectordb creation
        with open('app/test_questions.txt', 'r') as file: ## STEP 2 Read all questions
            for qstn in file:
                questions.append(qstn.strip())
                result = self.question_answer(qstn.strip()) ## STEP 3 Ask one by one and get answer and contexts
                answers.append(result['answer'])
                cont_cont = []
                for cont in result['context']: ## STEP 4 Loop over all context to get only context text
                    cont_cont.append(cont.page_content)
                contexts.append(cont_cont)
        return_dict['question'] = questions ## STEP 5 Setup the return JSON object
        return_dict['answer'] = answers
        return_dict['contexts'] = contexts
        return return_dict

    def vectorise_doc_unstruct(self, fnam:str)->bool:
        """
        Inputs : Full filename of a document
        Output : Returns True if success else False
        If unstruct library available calls document processing based on it
        Else calls document processing based on LangChain
        """
        if vec_method == 'UNSTRUCT':
            doc_file = 'app/documents.pkl'
            if os.path.exists(doc_file):
                with open(doc_file, 'rb') as file:
                    documents = pickle.load(file)
            else:
                documents = self.chunk_document(fnam)
                with open(doc_file, 'wb') as file:
                    pickle.dump(documents, file)
        elif vec_method == 'LANGC':
            documents = self.chunk_document_langchain(fnam)

        embeddings = self._embeddings
        try:
            self.vectorstore = Chroma.from_documents(documents, embeddings) 
            #not using persistent storage - reasonable to store in memory
        except Exception as e:
            raise Exception(f'Error in processing {fnam} {e}')
            
        return True


    def chunk_document(self, fnam:str)->list:
        """
        Inputs : Full filename of a document
        Output : Returns a list of document chunks each of type langchain_core.documents.Document
        This function takes a file (pdf file in this case) split into chunks (now use split by title considering the 
        structure of the eBook.pdf) and returns the list document chunks
        """
        documents = []
        metadata = {}
        metadata['filename'] = fnam #There is only one file so the filename-metadata added directly for simplicity
        try:
            elements = partition_pdf(fnam, strategy='hi_res', hi_res_model_name='yolox')
            #uses yolox a CV model to parse through the Ebook. The ebook also contains images,
            #we could use a image captioning model to get captions(summary) of the image - Dont have GPU
            #Since the doesnt have tables didnit  use df_infer_table_structure=True
            chunked_elements = chunk_by_title(elements)
            for element in chunked_elements:
                chpt = re.findall('CHAPTER[]*[0-9]+',element.text)
                metadata['filename'] = chpt[0] if len(chpt) > 0 else ''
                documents.append(Document(page_content=element.text, metadata=metadata))
        except Exception as e:
            raise Exception(f'Error in processing {fnam} {e}')
        return documents

    def chunk_document_langchain(self, fnam:str)->list:
        """
        This another implementation of chunk_document in case the unstructured is not available (may not be available for python 3.11)
        Inputs : Full filename of a document
        Output : Returns a list of document chunks each of type langchain_core.documents.Document
        This function takes a file (pdf file in this case) split into chunks (now use split by title considering the
        structure of the eBook.pdf) and returns the list document chunks
        """
        documents = []
        metadata = {}
        metadata["filename"] = fnam
        try:
            loader = PyPDFLoader(fnam)
            documents.extend(loader.load())
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
            chunked_documents = text_splitter.split_documents(documents)
        except Exception as e:
            raise Exception(f'Error in processing {fnam} {e}')
        return chunked_documents


    def question_answer(self, question:str):
        """
        Input : Question asked based on the document
        Output : Result from LLM (contains the question asked, answer generated and 
        the context retrieved and supplied to the LLM
        """
        retriever = self.vectorstore.as_retriever()
        retriever = self.set_retriever(question)
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know."
                "\n\n"
                "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
        )
        question_answer_chain = create_stuff_documents_chain(self._llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": question})

        return result

    def set_retriever(self, question:str):
        """
        Input : Question asked
        Output : return a context retriever 
        This functions builds and returns a retriever
        The retriever is an emseble of similary search and BM25 algorithm with equal weights
        """
        ret_chunks = self.vectorstore.similarity_search(question, k=15)
        keyword_retriever = BM25Retriever.from_documents(ret_chunks)
        vectorstore_retreiver = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15} #Retrieve 15 most similar chunks
        )
        ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                               keyword_retriever],
                                               weights=[0.5, 0.5]) #giving equal wight for both similarity and BM25 (can finetune if ground truth given)
        return ensemble_retriever
