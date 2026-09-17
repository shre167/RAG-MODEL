L> python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('GE_API_KEY=', os.getenv('GE_API_KEY')); print('EMBEDDING_MODEL=', os.getenv('EMBEDDING_MODEL'))"
GE_API_KEY= H9hr3caxZM81kC32ZMKZh8GLMudIArGC2NMDoTC5
EMBEDDING_MODEL= cohere.embed-english-v3
(.venv) PS C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL> git grep load_dotenv
error: cannot spawn less: No such file or directory
fatal: unable to execute pager 'less'
(.venv) PS C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL> python -m pip install python-dotenv
Requirement already satisfied: python-dotenv in c:\users\shrsamal\onedrive - capgemini\desktop\astronomy rag\rag-model\.venv\lib\site-packages (1.2.3)

[notice] A new release of pip is available: 25.3 -> 26.2.1
[notice] To update, run: python.exe -m pip install --upgrade pip
(.venv) PS C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL> python ingest.py                          
Batch embedding failed (Error code: 403 - {'Message': 'User is not authorized to access this resource with an explicit deny in an identity-based policy'}); retrying per-item.
OpenAI client embedding error (model=cohere.embed-english-v3): Error code: 403 - {'Message': 'User is not authorized to access this resource with an explicit deny in an identity-based policy'}
Ingestion error: Embedding generation failed during ingestion after persisting 0/898 missing chunks: Embedding failed for item 1/10
(.venv) PS C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL> python -c "from dotenv import load_dotenv; load_dotenv(); import os; from openai import OpenAI; client=OpenAI(api_key=os.getenv('GE_API_KEY'), base_url=os.getenv('LLM_BASE_URL')); print(client.models.list())"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from dotenv import load_dotenv; load_dotenv(); import os; from openai import OpenAI; client=OpenAI(api_key=os.getenv('GE_API_KEY'), base_url=os.getenv('LLM_BASE_URL')); print(client.models.list())
                                                                                                                                                                                   ~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL\.venv\Lib\site-packages\openai\resources\models.py", line 95, in list
    return self._get_api_list(
           ~~~~~~~~~~~~~~~~~~^
        "/models",
        ^^^^^^^^^^
    ...<8 lines>...
        model=Model,
        ^^^^^^^^^^^^
    )
    ^
  File "C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL\.venv\Lib\site-packages\openai\_base_client.py", line 1454, in get_api_list
    return self._request_api_list(model, page, opts)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL\.venv\Lib\site-packages\openai\_base_client.py", line 1253, in _request_api_list
    return self.request(page, options, stream=False)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL\.venv\Lib\site-packages\openai\_base_client.py", line 1141, in request
    raise self._make_status_error_from_response(err.response) from None
openai.PermissionDeniedError: Error code: 403 - {'Message': 'User is not authorized to access this resource with an explicit deny in an identity-based policy'}
(.venv) PS C:\Users\shrsamal\OneDrive - Capgemini\Desktop\Astronomy RAG\RAG-MODEL

LLM_BASE_URL=https://openai.generative.engine.capgemini.com/v1
LLM_MODEL=amazon.nova-lite-v1:0  
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
