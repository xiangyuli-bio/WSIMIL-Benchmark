"""Common public metadata for complete native-reference execution profiles."""
import copy
import json


def native_profile(profile, *, class_count):
    value = copy.deepcopy(profile)
    required = {'id', 'family', 'tasks', 'required_inputs', 'external_source', 'native_recipe'}
    if not required <= value.keys() or not isinstance(value['native_recipe'], dict) or not value['native_recipe']:
        raise ValueError('A native execution profile needs its full explicit native recipe and common fields')
    if class_count is not None and (type(class_count) is not int or class_count < 2):
        raise ValueError('Declare a positive native class count or an explicit variable-class interface')
    if 'class_count' in value and value['class_count'] != class_count:
        raise ValueError('Common class metadata cannot override an existing native execution contract')
    value['class_count'] = class_count
    if 'recipe' not in value:
        value['recipe'] = ('Complete ' + value['family'] + ' reference [' + value['id'] + ']; native_recipe=' +
            json.dumps(value['native_recipe'], sort_keys=True, separators=(',', ':')))
    if not isinstance(value['recipe'], str) or not value['recipe'].strip():
        raise ValueError('Every native execution profile needs a nonempty public recipe description')
    return value
