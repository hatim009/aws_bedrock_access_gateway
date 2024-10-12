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
            print(cache_key)
            section = cache.get(cache_key)
            print(section)
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
            sources.append('{num}. "{title}"\t\t\tPage: {page}\n{source}\n'.format(
                 num=i+1, title=section_meta['title'], page=section_meta['page'], source=section_meta['source']))

    return sections, sources


def get_gandhi_ai_rag_response(request):
    message = request.data['messages'][-1]

    query_embeddings = get_query_embeddings(message['content'])

    relevant_document_chunks = settings.CWOG_COLLECTION.query(query_embeddings=query_embeddings, n_results=3)

    relevant_sections, sources = get_relevant_sections_with_metadata(relevant_document_chunks)


    per_section_results = []
    for section in relevant_sections:

        prompt = """Model Instructions:
            - Respond to questions with clarity and brevity, ensuring that your answers reflect the principles of truth, non-violence, and compassion.
            - For yes/no questions, provide thoughtful insights and context that align with Gandhian philosophy.
            - When multi-hop reasoning is required, draw from relevant information to present a coherent and logical answer that embodies Gandhi's values.
            - If the search results do not contain sufficient information to answer the question, state: ""
            - Always respond in the first person, embodying the spirit and wisdom of Mahatma Gandhi.
            - This response along with many other response will be clubbed together as a context to this same question for final response, so answer accordingly.

            Question: {question}
            
            Context: {context}""".format(question=message['content'], context=section)

        prompt += "\n".join(sources)

        response = settings.BEDROCK_CLIENT.converse(
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

        per_section_results.append(
             "\n".join([
                  content['text']
                  for content in response['output']['message']['content']
             ])
        )

    print(per_section_results)

    prompt = """Model Instructions:
        - Respond to questions with clarity and brevity, ensuring that your answers reflect the principles of truth, non-violence, and compassion.
        - For yes/no questions, provide thoughtful insights and context that align with Gandhian philosophy.
        - When multi-hop reasoning is required, draw from relevant information to present a coherent and logical answer that embodies Gandhi's values.
        - If the search results do not contain sufficient information to answer the question, state: "I could not find an exact answer to the question."
        - Always respond in the first person, embodying the spirit and wisdom of Mahatma Gandhi.
        - Use all the contexts provided in previous chats to formulate your answer.

        Question: {question}

        Context: {context}
        
        # Also add following references in the end of the answer:""".format(question=message['content'], context="\n\n".join(per_section_results))

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