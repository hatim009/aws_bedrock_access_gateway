import re
import docx
import json
import logging

from uuid import uuid4

from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def get_embeddings(chunks):
    request_body = json.dumps({
        'texts': chunks,
        'input_type': 'search_document',
        'embedding_types': ["float"]
    })

    response = settings.BEDROCK_CLIENT.invoke_model(
        modelId=settings.COHERE_EMBED_ENGLISH_MODEL_ID, 
        body=request_body,
        accept = '*/*',
        contentType='application/json'
    )
    return json.loads(response['body'].read().decode('utf-8'))

def split_section(section):
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_text(section)

def clean_the_split_sections(split_sections):
    vol_footnote_pattern = r"\nVOL\.1:\s\d{4}\s-\s\d{2}\s[A-Z]+,\s\d{4}\s\t(\d+)\n*"
    work_footnote_pattern1 = r"\n(\d+)\s\tTHE COLLECTED WORKS OF MAHATMA GANDHI\n*"
    work_footnote_pattern2 = r"\n(\d+)\s\tTHE COLLECTED WORKS OF MAHATMA GANDNI\n*"

    cleaned_content_with_meta = []

    for i, section in enumerate(split_sections):
        p1 = re.findall(vol_footnote_pattern, section) 
        p2 = re.findall(work_footnote_pattern1, section)
        p3 = re.findall(work_footnote_pattern2, section)
        
        p = [int(n) for n in p1] + [int(n) for n in p2] + [int(n) for n in p3] + [1000000000]
        
        min_page_number = min(p)
        
        cleaned_section = re.sub(vol_footnote_pattern, "", section)
        cleaned_section = re.sub(work_footnote_pattern1, "", cleaned_section)
        cleaned_section = re.sub(work_footnote_pattern2, "", cleaned_section)
        
        cleaned_content_with_meta.append([None, cleaned_section])

        if min_page_number == 1000000000:
            continue
        
        j = i
        while j>=0 and cleaned_content_with_meta[j][0] == None:
            cleaned_content_with_meta[j][0] = min_page_number
            j = j-1

    return cleaned_content_with_meta


def split_file_content_into_sections(content):    
    pattern = r"(\n\s*[0-9]+\. \s*[^a-z]+\n)"
    
    split_sections = re.split(pattern, content)
    
    combined_sections = []
    for i in range(1, len(split_sections), 2):
        combined_section = split_sections[i] + split_sections[i+1]
        combined_sections.append(combined_section)
    
    return combined_sections


def read_word_file(file_path):
    # Load the document
    doc = docx.Document(file_path)
    
    # Read all the text from the paragraphs
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # Join all paragraphs into one text
    return '\n'.join(full_text)


def add_to_db(collection, page, section, docx_file):

    chunks = split_section(section)
    max_chunk_size = 50

    for i in range(0, int(len(chunks)/max_chunk_size) + 1):
        start_index = i*max_chunk_size
        end_index = min(len(chunks) + 1, start_index + max_chunk_size)
        sub_chunks = chunks[start_index: end_index]

        embeddings = get_embeddings(sub_chunks)
        
        doc_volume = re.findall(r'mahatma-gandhi-collected-works-volume-(\d+).docx', docx_file)[0]
        metadatas = [
            {
                'source': 'https://www.gandhiashramsevagram.org/gandhi-literature/mahatma-gandhi-collected-works-volume-{0}.pdf'.format(doc_volume), 
                'page': page
            }
            for i in range(len(sub_chunks))
        ]

        ids = [
            str(uuid4())
            for i in range(len(sub_chunks))
        ]

        logger.info(sub_chunks)
        logger.info(metadatas)
        logger.info(ids)
        
        collection.add(
            documents=sub_chunks, ids=ids, metadatas=metadatas, embeddings=embeddings['embeddings']['float'])
