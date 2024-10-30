import re
import json

from django.conf import settings
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor

from .decorators import botocore_backoff


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
    title_pattern = r"(\n\s*[0-9]+\. \s*[^a-z]+\n)"

    sections = []
    sections_meta = []
    visited_keys = set()
    for i in range(len(documents)):
        source_pattern = r'https://www.gandhiashramsevagram.org/gandhi-literature/mahatma-gandhi-collected-works-volume-(\d+).pdf'
        doc_vol = re.findall(source_pattern, metadatas[i]['source'])[0]
        doc_section = metadatas[i]['section']

        cache_key = settings.CWOG_CACHE_KEY_FORMAT.format(vol=doc_vol, section=doc_section)

        if cache_key not in visited_keys:
            section = cache.get(cache_key)
            title = re.split(title_pattern, section)[0].strip()

            sections.append(section)
            sections_meta.append({
                'title': title,
                'page': metadatas[i]['page'], 
                'source': metadatas[i]['source'],
            })

            visited_keys.add(cache_key)
    
    sources = []
    for i, section_meta in enumerate(sections_meta):
            sources.append('''{num}. "{title}"     Page: {page}
                           {source}'''.format(
                 num=i+1, title=section_meta['title'], page=section_meta['page'], source=section_meta['source']))

    return sections, sources


@botocore_backoff(retries=3)
def bedrock_converse(messages):
    return settings.BEDROCK_CLIENT.converse(
        modelId=settings.LLAMA_MODEL_ID,
        messages=messages,
        inferenceConfig={
            'temperature': 0.5
        }
    )

def concurrent_bedrock_converse(sections, max_workers=1):
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
         results = executor.map(bedrock_converse, sections)

    return results


def get_gandhi_ai_rag_response(request):
    message = request.data['messages'][-1]

    query_embeddings = get_query_embeddings(message['content'])

    relevant_document_chunks = settings.CWOG_COLLECTION.query(query_embeddings=query_embeddings, n_results=3)

    relevant_sections, sources = get_relevant_sections_with_metadata(relevant_document_chunks)

    request_sections = []
    PER_SECTION_CHAR_LIMIT = 30000
    for section in relevant_sections:

        prompt = """Model Instructions:
            - Respond to questions with clarity and brevity, ensuring that your answers reflect the principles of truth, non-violence, and compassion.
            - For yes/no questions, provide thoughtful insights and context that align with Gandhian philosophy.
            - When multi-hop reasoning is required, draw from relevant information to present a coherent and logical answer that embodies Gandhi's values.
            - If the search results do not contain sufficient information to answer the question, state: ""
            - Always respond in the first person, embodying the spirit and wisdom of Mahatma Gandhi.
            - This response along with many other response will be given together as a context to you for this very same question for final response, so answer accordingly.

            Question: {question}
            
            Context: {context}"""
        
        request_sub_sections = []
        for i in range(0, len(section), PER_SECTION_CHAR_LIMIT):
             sub_section = section[i:i+PER_SECTION_CHAR_LIMIT]
             sub_section_prompt = prompt.format(question=message['content'], context=sub_section)
             request_sub_sections.append([{
                'role': message['role'], 
                'content': [
                    {
                        'text': sub_section_prompt
                    }
                ]
            }])

        request_sections += request_sub_sections

    per_section_responses = concurrent_bedrock_converse(request_sections)

    full_context = "\n\n".join([
         (
            "\n".join([
                content['text']
                for content in response['output']['message']['content']
            ])
        ) for response in per_section_responses
    ])


    prompt = """Model Instructions:
        - Respond to questions with clarity and brevity, ensuring that your answers reflect the principles of truth, non-violence, and compassion.
        - For yes/no questions, provide thoughtful insights and context that align with Gandhian philosophy.
        - When multi-hop reasoning is required, draw from relevant information to present a coherent and logical answer that embodies Gandhi's values.
        - If the search results do not contain sufficient information to answer the question, state: "I could not find an exact answer to the question."
        - Always respond in the first person, embodying the spirit and wisdom of Mahatma Gandhi.
        - Use all the contexts provided in previous chats to formulate your answer.

        Question: {question}

        Context: {context}
        
        # Also add following references in the end of the answer:""".format(question=message['content'], context=full_context)

    prompt += "\n".join(sources)

    return settings.BEDROCK_CLIENT.converse_stream(
        modelId=settings.LLAMA_MODEL_ID,
        messages=[{
            'role': message['role'], 
            'content': [
                {
                    'text': prompt
                }
            ]
        }],
        inferenceConfig={
            'temperature': 0.5
        }
    )