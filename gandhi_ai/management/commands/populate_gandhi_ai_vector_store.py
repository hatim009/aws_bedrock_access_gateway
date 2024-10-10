import re
import os
import sys
import chromadb
import logging

from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError

from gandhi_ai.utils import (read_word_file, split_section, get_embeddings, 
                             clean_the_split_sections, split_file_content_into_sections)


logger = logging.getLogger(__name__)


class Command(BaseCommand):

    def add_to_db(self, collection, page, section, docx_file):

        chunks = split_section(section)
        max_chunk_size = 50

        for i in range(0, int(len(chunks)/max_chunk_size) + 1):
            start_index = i*max_chunk_size
            end_index = min(len(chunks) + 1, start_index + max_chunk_size)
            sub_chunks = [chunk for chunk in chunks[start_index: end_index] if re.findall(r"[a-zA-Z]", chunk)]

            if not sub_chunks:
                continue

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
            
            collection.add(
                documents=sub_chunks, ids=ids, metadatas=metadatas, embeddings=embeddings['embeddings']['float'])


    def handle(self, *args, **options):
        try:
            client = chromadb.PersistentClient(path="./gandhi_ai_vector_store")
            try:
                print("Delete collected_works_of_gandhi collection if exist.")
                client.delete_collection('collected_works_of_gandhi')
            except Exception as e:
                print("Collection collected_works_of_gandhi does not exist, creating one.")
            
            collection = client.get_or_create_collection('collected_works_of_gandhi', metadata={"hnsw:space": "cosine"})

            collected_works_of_gandhi = os.listdir('./resources/collected_works_of_gandhi')
            
            embedded_cwog_files = []
            try:
                with open('./resources/embedded_cwog_files.txt', 'r') as fp:
                    embedded_cwog_files = fp.read().split('\n')
            except Exception as e:
                pass

            for docx_file in collected_works_of_gandhi:
                print(docx_file)
                if docx_file in embedded_cwog_files:
                    print("File already populated, skipping...")
                    continue
                file_path = "./resources/collected_works_of_gandhi/{0}".format(docx_file)
                content = read_word_file(file_path)
                split_sections = split_file_content_into_sections(content)
                cleaned_sections_with_meta = clean_the_split_sections(split_sections)
                for cleaned_section in cleaned_sections_with_meta:
                    self.add_to_db(collection, cleaned_section[0], cleaned_section[1], docx_file) 

                with open('./resources/embedded_cwog_files.txt', 'a') as fp:
                    fp.write(docx_file)
        except RuntimeError:
            raise CommandError('Error populating embeddings for collected_works_of_gandhi DB.').with_traceback(sys.exception().__traceback__)

        self.stdout.write(
            self.style.SUCCESS('Sucessfully populated embeddings for collected_works_of_gandhi DB.')
        )