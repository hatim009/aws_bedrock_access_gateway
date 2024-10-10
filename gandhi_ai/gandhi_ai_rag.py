import re
import json

from django.conf import settings
from django.core.cache import cache


def get_query_embeddings(query_text):
    request_body = json.dumps({
        'texts': [query_text],
        'input_type': 'search_query',
        'embedding_types': ["float"]
    })

    response = settings.BEDROCK_CLIENT.invoke_model(
        modelId=settings.COHERE_EMBED_ENGLISH_MODEL_ID, 
        body=request_body,
        accept = '*/*',
        contentType='application/json'
    )
    return json.loads(response['body'].read().decode('utf-8'))['embeddings']['float'] 


def get_relevant_sections_with_metadata(relevant_document_chunks):
    metadatas = relevant_document_chunks['metadatas'][0]
    documents = relevant_document_chunks['documents'][0]

    sections = []
    visited_keys = set()
    for i in range(len(documents)):
        source_pattern = r'https://www.gandhiashramsevagram.org/gandhi-literature/mahatma-gandhi-collected-works-volume-(\d+).pdf'
        doc_vol = re.findall(source_pattern, metadatas[i]['source'])
        doc_section = metadatas[i]['section']

        cache_key = settings.CWOG_CACHE_KEY_FORMAT.format(vol=doc_vol, section=doc_section)

        if cache_key not in visited_keys:
            sections.add({
                'section': cache.get(cache_key),
                'metadata': metadatas[i]
            })
            visited_keys.add(cache_key)

    return sections


def get_gandhi_ai_rag_response(request):
    messages = request.data['messages']

    query_embeddings = get_query_embeddings(messages['content'])

    relevant_document_chunks = settings.CWOG_COLLECTION.query(query_embeddings=query_embeddings, n_results=10)

    relevant_sections, metadata = get_relevant_sections_with_metadata(relevant_document_chunks)

    relevant_sections_combined = "\n\n\n".join(relevant_sections)

    prompt = """Model Instructions:\n""" 
    + """•⁠ ⁠You should provide concise answer to simple questions when the answer is directly contained in search results, but when comes to yes/no question, provide some details.\n"""
    + """•⁠ ⁠In case the question requires multi-hop reasoning, you should find relevant information from search results and summarize the answer based on relevant information with logical reasoning.\n"""
    + """•⁠ ⁠If the search results do not contain information that can answer the question, please state that "I could not find an exact answer to the question.\n\n"""
    + """Question: {question} \nContext: {context} \nAnswer:"""

    return settings.BEDROCK_CLIENT.converse_stream(
        modelId=settings.LLAMA_MODEL_ID,
        messages=[
            {
                'role': message['role'], 
                'content': [
                    {
                        'text': message['content']
                    }
                ]
            }
            for message in messages
        ]
    )