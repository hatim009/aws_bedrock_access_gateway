import os

from django.apps import AppConfig
from django.core.cache import cache
from django.conf import settings

from gandhi_ai.utils import (read_word_file, split_file_content_into_sections, clean_the_split_sections)


class GandhiAiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gandhi_ai'

    def ready(self):
        # This method runs on service startup
        self.load_sections_from_cwog()

    def load_sections_from_cwog(self):
        collected_works_of_gandhi = os.listdir('./resources/collected_works_of_gandhi')

        if not settings.SKIP_CWOG_CACHE_CREATION:
            for docx_file in collected_works_of_gandhi:
                file_path = "./resources/collected_works_of_gandhi/{0}".format(docx_file)
                content = read_word_file(file_path)
                split_sections = split_file_content_into_sections(content)
                cleaned_sections_with_meta = clean_the_split_sections(split_sections)
