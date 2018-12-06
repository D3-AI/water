# -*- coding: utf-8 -*-

import json
import os

import pandas as pd


class TimeSeriesLoader(object):

    def __init__(self, dataset_path, target_table, target_column=None):
        self._dataset_path = dataset_path
        self._target_table = target_table
        self._target_column = target_column

        metadata_path = os.path.join(dataset_path, 'metadata.json')
        with open(metadata_path, 'r') as metadata_file:
            self._metadata = json.load(metadata_file)

    def _load_entities(self):
        entities = dict()
        relationships = list()

        for table in self._metadata['tables']:
            table_path = os.path.join(self._dataset_path, table['path'])
            data = pd.read_csv(table_path)
            table_name = table['name']
            index = table['primary_key']
            time_index = None

            for field in table['fields']:
                field_name = field['name']
                field_type = field['type']
                field_subtype = field.get('subtype')

                if field_type == 'datetime':
                    if field_subtype == 'time_index':
                        if time_index is not None:
                            raise ValueError(
                                "More than one time_index found in table {}".format(table_name)
                            )

                        time_index = field_name

                    column = data[field_name]
                    dt_format = field['properties']['format']
                    data[field_name] = pd.to_datetime(column, format=dt_format)

                elif field_type == 'id' and field_subtype == 'foreign':
                    foreign = field['ref']
                    relationships.append((
                        foreign['table'],
                        foreign['field'],
                        table_name,
                        index
                    ))

            table_details = [data, index]
            if time_index is not None:
                table_details.append(time_index)

            entities[table_name] = tuple(table_details)

        return entities, relationships

    def load(self, target=True):
        entities, relationships = self._load_entities()

        data = {
            'entities': entities,
            'relationships': relationships,
            'target_entity': self._target_table,
            'target_column': self._target_column,
            'dataset_name': self._metadata['name']
        }

        X = entities[self._target_table][0]

        if target:
            y = X.pop(self._target_column)
            return X, y, data

        else:
            return X, data
