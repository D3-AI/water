# -*- coding: utf-8 -*-

import getpass
import json
import logging

from pymongo import MongoClient

from water.utils import restore_dots

LOGGER = logging.getLogger(__name__)


def MongoDB(object):

    def __init__(self, database=None, config=None, **kwargs):
        if config:
            with open(config, 'r') as f:
                config = json.load(f)
        else:
            config = kwargs

        host = config.get('host', 'localhost')
        port = config.get('port', 27017)
        user = config.get('user')
        password = config.get('password')
        database = database or config.get('database', 'test')
        auth_database = config.get('auth_database', 'admin')

        if user and not password:
            password = getpass.getpass(prompt='Please insert database password: ')

        client = MongoClient(
            host=host,
            port=port,
            username=user,
            password=password,
            authSource=auth_database
        )

        LOGGER.info("Setting up a MongoClient %s", client)

        self.db = client[database]

    def load_template(self, template_name):
        match = {
            'name': template_name
        }
        project = {
            '_id': 0
        }

        cursor = self.db.templates.find(match, project)
        templates = list(cursor.sort('insert_ts', -1).limit(1))

        if templates:
            return restore_dots(templates[0])
