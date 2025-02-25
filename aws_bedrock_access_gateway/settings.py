"""
Django settings for aws_bedrock_access_gateway project.

Generated by 'django-admin startproject' using Django 4.2.5.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""
import os
import boto3
import chromadb
import google.generativeai as genai

from pathlib import Path
from decouple import config

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-fyc2(fx&b!fl4f0ci(8%gpp-+(^_*=q3n-^fl56756l&=&_*5s'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = (config('DEBUG') == 'True')

SKIP_CWOG_CACHE_CREATION = False

ALLOWED_HOSTS = ['*']

CSRF_TRUSTED_ORIGINS = ['https://gandhitoday.cloudrunr.co']

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'gandhi_ai'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'aws_bedrock_access_gateway.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'aws_bedrock_access_gateway.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Cache

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6389/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# LOGGING CONFIGURATION
LOG_PATH = os.path.join(BASE_DIR, "var/log/aws_bedrock_access_gateway/")
log_path = Path(LOG_PATH)
log_path.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,  # the dictConfig format version
    "disable_existing_loggers": False,  # retain the default loggers
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(LOG_PATH, "application.log"),
            'maxBytes': 1024*1024*5, # 5 MB
            'backupCount': 5,
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["file"],
        },
    },
}


BEDROCK_CLIENT = boto3.client(
    'bedrock-runtime', region_name='us-east-1', 
    aws_access_key_id=config('AWS_ACCESS_KEY'), aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY')
)

LLAMA_MODEL_ID = 'meta.llama3-8b-instruct-v1:0'

COHERE_EMBED_ENGLISH_MODEL_ID = 'cohere.embed-english-v3'


CHROMA_DB_CLIENT = chromadb.PersistentClient(path="./gandhi_ai_vector_store")

CWOG_COLLECTION_NAME = 'collected_works_of_gandhi'

CWOG_COLLECTION = CHROMA_DB_CLIENT.get_collection(CWOG_COLLECTION_NAME)

CWOG_CACHE_KEY_FORMAT = "vol:{vol}-section:{section}"


genai.configure(api_key=config("GOOGLE_GEMINI_API_KEY"))
GENAI_MODEL = genai.GenerativeModel("gemini-1.5-flash")