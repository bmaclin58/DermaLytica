"""
Django settings for DermaLytica project.

Generated by 'django-admin startproject' using Django 5.1.6.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""
import os
from pathlib import Path
import environ
from dotenv import load_dotenv
from urllib.parse import urlparse


# ------------------------------------------------------------------------
#                 .Env Secret Variables / Path Building
# ------------------------------------------------------------------------

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent


def get_secret(secret_name):
	from google.cloud import secretmanager
	"""Fetches a secret value from GCP Secret Manager."""
	project_id = "143642567909"  # Replace with your actual GCP project ID
	client = secretmanager.SecretManagerServiceClient()

	name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

	try:
		response = client.access_secret_version(name = name)
		return response.payload.data.decode("UTF-8")
	except Exception as e:
		print(f"Error retrieving secret {secret_name}: {e}")
		return None  # Ensure application doesn't crash

env_file = os.path.join(BASE_DIR, ".env")

if os.path.isfile(env_file):
	# Use a local secret file, if provided
	load_dotenv()
	SECRET_KEY = os.getenv("DJANGO_KEY")
	YELP_API_KEY = os.getenv("YELP_API_KEY")

else:
	# Pull secrets from Secret Manager
	SECRET_KEY = get_secret("djangoSettings")
	YELP_API_KEY = get_secret("YELP_API_KEY")

# Fetch secrets
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DEBUG = True

ALLOWED_HOSTS = ['dermalytica-143642567909.us-central1.run.app', 'hirebrianmaclin.com']
CSRF_TRUSTED_ORIGINS = ['dermalytica-143642567909.us-central1.run.app','hirebrianmaclin.com']


# Application definition

INSTALLED_APPS = [
		'whitenoise.runserver_nostatic',
		'crispy_forms',
		"crispy_bootstrap5",
		'django.contrib.admin',
		'django.contrib.auth',
		'django.contrib.contenttypes',
		'django.contrib.sessions',
		'django.contrib.messages',
		'django.contrib.staticfiles',
		'DermaLytica.apps.DermaLyticaConfig',
		'Home_Portfolio.apps.HomePortfolioConfig'
		]

MIDDLEWARE = [
		'django.middleware.security.SecurityMiddleware',
		'whitenoise.middleware.WhiteNoiseMiddleware',
		'django.contrib.sessions.middleware.SessionMiddleware',
		'django.middleware.common.CommonMiddleware',
		#'django.middleware.csrf.CsrfViewMiddleware',
		'django.contrib.auth.middleware.AuthenticationMiddleware',
		'django.contrib.messages.middleware.MessageMiddleware',
		'django.middleware.clickjacking.XFrameOptionsMiddleware',
		]

ROOT_URLCONF = 'urls'

TEMPLATES = [
		{
				'BACKEND':  'django.template.backends.django.DjangoTemplates',
				'DIRS':     [BASE_DIR / 'templates'],
				'APP_DIRS': True,
				'OPTIONS':  {
						'context_processors': [
								'django.template.context_processors.debug',
								'django.template.context_processors.request',
								'django.contrib.auth.context_processors.auth',
								'django.contrib.messages.context_processors.messages',
								],
						},
				},
		]

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

WSGI_APPLICATION = 'wsgi.application'

# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

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
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = "America/New_York"

USE_I18N = True

USE_TZ = True

# ------------------------------------------------------------------------
#          Static / Media files (CSS, JavaScript, Images, Docs)
#        https://docs.djangoproject.com/en/5.1/howto/static-files/
# ------------------------------------------------------------------------

STATIC_URL = "/staticfiles/"
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles/')
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/Files/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
