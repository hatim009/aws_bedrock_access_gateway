import re
import os
import sys
import logging


from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.core.cache import cache

from gandhi_ai.utils import (read_word_file, clean_the_split_sections, split_file_content_into_sections)


logger = logging.getLogger(__name__)


class Command(BaseCommand):

    def handle(self, *args, **options):
        try:
            collected_works_of_gandhi = os.listdir('./resources/collected_works_of_gandhi')
            if not settings.SKIP_CWOG_CACHE_CREATION:
                for docx_file in collected_works_of_gandhi:
                    file_path = "./resources/collected_works_of_gandhi/{0}".format(docx_file)
                    print(file_path)
                    content = read_word_file(file_path)
                    split_sections = split_file_content_into_sections(content)
                    pages, cleaned_sections = clean_the_split_sections(split_sections)
                    
                    vol = re.findall(r'mahatma-gandhi-collected-works-volume-(\d+).docx', docx_file)[0]
                    for section, cleaned_section in enumerate(cleaned_sections):
                        key = settings.CWOG_CACHE_KEY_FORMAT.format(vol=vol, section=section)
                        value = cleaned_section
                        cache.set(key, value, timeout=None)
        except RuntimeError:
            raise CommandError('Error populating redis cache with CWOG sections.').with_traceback(sys.exception().__traceback__)

        self.stdout.write(
            self.style.SUCCESS('Sucessfully populated redis cache with CWOG sections.')
        )