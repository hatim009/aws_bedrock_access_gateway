import re
import json
import docx

from django.conf import settings

from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_word_file(file_path):
    # Load the document
    doc = docx.Document(file_path)
    
    # Read all the text from the paragraphs
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # Join all paragraphs into one text
    return '\n'.join(full_text)


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
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=250)
    return text_splitter.split_text(section)



def clean_the_split_sections(split_sections):
    vol_footnote_pattern1 = r"\n*VOL\.\s*\d+\s*:\s*\d{4}\s*-\s*\d+\s*[A-Z]+\s*,\s*\d{4}\s*\.*\s*\t*(\d+)\n*"
    vol_footnote_pattern2 = r"\n*VOL\.\s*\d+\s*:\s*\d+\s*[A-Z]+\s*,\s*\d{4}\s*-\s*\d+\s*[A-Z]+\s*,\s*\d{4}\s*\.*\s*\t*(\d+)\n*"
    work_footnote_pattern1 = r"\n*(\d+)\s*\t*THE COLLECTED WORKS OF MAHATMA GANDHI\n*"
    work_footnote_pattern2 = r"\n*(\d+)\s*\t*THE COLLECTED WORKS OF MAHATMA GANDNI\n*"

    pages = []
    cleaned_sections = []

    for i, section in enumerate(split_sections):
        p1 = re.findall(vol_footnote_pattern1, section) 
        p2 = re.findall(vol_footnote_pattern2, section) 
        p3 = re.findall(work_footnote_pattern1, section)
        p4 = re.findall(work_footnote_pattern2, section)
        
        p = [int(n) for n in p1] + [int(n) for n in p2] + [int(n) for n in p3] + [int(n) for n in p4] + [1000000000]
        
        min_page_number = min(p)
        
        cleaned_section = re.sub(vol_footnote_pattern1, "", section)
        cleaned_section = re.sub(vol_footnote_pattern2, "", cleaned_section)
        cleaned_section = re.sub(work_footnote_pattern1, "", cleaned_section)
        cleaned_section = re.sub(work_footnote_pattern2, "", cleaned_section)
        
        pages.append(None)
        cleaned_sections.append(cleaned_section)

        if min_page_number == 1000000000:
            continue
        
        j = i
        while j>=0 and pages[j] == None:
            pages[j] = min_page_number
            j = j-1

    return pages, cleaned_sections


def split_file_content_into_sections(content):    
    pattern = r"(?:\n\s*[0-9]+\. \s*[^a-z]+\n)|(?:\nCHAPTER [IVXLCDM]+\n)|(?:\nAPPENDIX [IVXLCDM]+\n)"
    
    split_sections = re.split(pattern, content)
    
    combined_sections = []
    for i in range(1, len(split_sections), 2):
        combined_section = split_sections[i] + split_sections[i+1]
        combined_sections.append(combined_section)
    
    return combined_sections