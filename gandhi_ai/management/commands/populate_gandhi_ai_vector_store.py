import os
import sys
import chromadb

from django.core.management.base import BaseCommand, CommandError

from gandhi_ai.management.utils.gandhi_ai_vector_store_utils import add_to_db, read_word_file, split_file_content_into_sections, clean_the_split_sections


class Command(BaseCommand):

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

            for docx_file in collected_works_of_gandhi:
                print(docx_file)
                file_path = "./resources/collected_works_of_gandhi/{0}".format(docx_file)
                content = read_word_file(file_path)
                split_sections = split_file_content_into_sections(content)
                cleaned_sections_with_meta = clean_the_split_sections(split_sections)
                for cleaned_section in cleaned_sections_with_meta:
                    add_to_db(collection, cleaned_section[0], cleaned_section[1], docx_file) 

        except RuntimeError:
            raise CommandError('Error populating embeddings for collected_works_of_gandhi DB.').with_traceback(sys.exception().__traceback__)

        self.stdout.write(
            self.style.SUCCESS('Sucessfully populated embeddings for collected_works_of_gandhi DB.')
        )