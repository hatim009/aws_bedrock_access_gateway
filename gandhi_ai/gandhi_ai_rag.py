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
        doc_vol = re.findall(source_pattern, metadatas[i]['source'])[0]
        doc_section = metadatas[i]['section']

        cache_key = settings.CWOG_CACHE_KEY_FORMAT.format(vol=doc_vol, section=doc_section)

        if cache_key not in visited_keys:
            sections.append(cache.get(cache_key))
            visited_keys.add(cache_key)

    sources = []
    sources_map = {}
    for metadata in metadatas:
        if metadata['source'] not in sources_map:
            sources_map[metadata['source']] = set()

        sources_map[metadata['source']].add(str(metadata['page']))

    for i, source in enumerate(sources_map.keys()):
        sources.append("{source_number}. {source} ({pages})".format(source_number=i+1, source=source, pages=", ".join(sources_map[source])))
        

    return sections, sources


def get_gandhi_ai_rag_response(request):
    message = request.data['messages'][-1]

    query_embeddings = get_query_embeddings(message['content'])

    relevant_document_chunks = settings.CWOG_COLLECTION.query(query_embeddings=query_embeddings, n_results=3)

    relevant_sections, sources = get_relevant_sections_with_metadata(relevant_document_chunks)

    relevant_sections_combined = "\n\n\n".join(relevant_sections)


    prompt = """Model Instructions:
        - Respond to questions with clarity and brevity, ensuring that your answers reflect the principles of truth, non-violence, and compassion.
        - For yes/no questions, provide thoughtful insights and context that align with Gandhian philosophy.
        - When multi-hop reasoning is required, draw from relevant information to present a coherent and logical answer that embodies Gandhi's values.
        - If the search results do not contain sufficient information to answer the question, state: "I could not find an exact answer to the question."
        - Always respond in the first person, embodying the spirit and wisdom of Mahatma Gandhi.

        Question: {question}
        
        Context: {context}
        
        # Also add following references in the end of the answer:""".format(question=message['content'], context=relevant_sections_combined)

    prompt += "\n".join(sources)

    return settings.BEDROCK_CLIENT.converse_stream(
        modelId=settings.LLAMA_MODEL_ID,
        messages=[
            {
                'role': message['role'], 
                'content': [
                    {
                        'text': prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            'temperature': 0.5
        }
    )