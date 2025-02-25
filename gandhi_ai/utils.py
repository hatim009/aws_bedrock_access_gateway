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
    for counter, para in enumerate(doc.paragraphs):
        if para.style.name.startswith('Heading') and not re.findall(r'[0-1]\s*.', para.text):
            full_text.append("\n" + str(counter+1) + ". " + para.text + "\n")
        else:
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
    vol_footnote_pattern1 = r"\n*VOL\.\s*\d+\s*:\s*\d{4}\s*-\s*\d+\s*[A-Z]+\s*,*\s*\d{4}\s*\.*\s*\t*(\d+)\n*"
    vol_footnote_pattern2 = r"\n*VOL\.\s*\d+\s*:\s*\d+\s*[A-Z]+\s*,*\s*\d{4}\s*-\s*\d+\s*[A-Z]+\s*,*\s*\d{4}\s*\.*\s*\t*(\d+)\n*"
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


# Define recursive function for splitting
def recursive_split(text, patterns):
    if not patterns:
        return [text]  # Base case: If no patterns left, return the text as a single item.
    
    pattern = patterns[0]  # Take the first pattern.
    result = []
    
    # Split text using the current pattern
    split_text = re.split(pattern, text)
    
    # For each part, recursively apply the remaining patterns
    for part in split_text:
        if part:  # Skip empty strings
            result.extend(recursive_split(part, patterns[1:]))  # Recurse on the remaining patterns.
    
    return result

def split_file_content_into_sections(content):
    patterns = [
        r"(\n1. SPEECH AT WORKING COMMITTEE MEETING, )",
        r"(\n\s*[0-9]+\s*\.\s*[^a-z]+\s*\n)",
        r"(\n\s*CHAPTER [IVXLCDM]+\s*\n)",
        r"(\n\s*APPENDIX [IVXLCDM]+\s*\n)"
    ]
    
    split_sections = recursive_split(content, patterns)

    combined_sections = []
    for i in range(0, len(split_sections), 2):
        combined = split_sections[i] + split_sections[i+1]

        combined_sections.append(combined)

    return combined_sections