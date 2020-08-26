#! /usr/bin/python3
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/pneumonia_detection_api/app/")

from client import app as application
application.secret_key = "SecretSessionKey"
