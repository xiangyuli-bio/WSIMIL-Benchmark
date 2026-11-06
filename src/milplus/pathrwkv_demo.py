"""Synthetic PathRWKV classification, regression, survival and mixed targets."""
import pandas as pd
import yaml

from .demo import create_demo


def create_pathrwkv_demo(output, dimension=16, patients=20, task_type='multitask'):
    if task_type not in {'classification', 'regression', 'survival', 'multitask'}:
        raise ValueError('Unknown PathRWKV synthetic task')
    root = create_demo(output, patients=patients, dimension=dimension,
                       survival=task_type == 'survival', regression=task_type == 'regression')
    path = root/'task.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['training_unit'] = 'slide'
    cfg['encoders']['toy']['spatial']['coordinate_order'] = 'xy'
    if task_type == 'multitask':
        rows = pd.read_csv(root/'slides.csv')
        targets = []
        for row in rows.to_dict('records'):
            patient = int(row['patient_id'].rsplit('_', 1)[1])
            targets.append({'slide_id': row['slide_id'], 'disease': row['label'],
                'age': '' if patient % 4 == 0 else 40+patient,
                'OS_time': '' if patient % 5 == 0 else row['time'],
                'OS_event': '' if patient % 5 == 0 else row['event']})
        pd.DataFrame(targets).to_csv(root/'targets.csv', index=False)
        cfg.update(task_type='multitask', classes=[], task_id='synthetic_pathrwkv_multitask')
        cfg['multitask'] = {'schema_version': 1, 'table': 'targets.csv', 'tasks': [
            {'name': 'disease', 'type': 'classification', 'column': 'disease',
             'classes': ['class_0', 'class_1'], 'unit': 'patient', 'label_provenance': 'synthetic only'},
            {'name': 'age', 'type': 'regression', 'column': 'age', 'unit': 'patient',
             'label_provenance': 'synthetic only'},
            {'name': 'OS', 'type': 'survival', 'time_column': 'OS_time',
             'event_column': 'OS_event', 'unit': 'patient', 'time_unit': 'months',
             'label_provenance': 'synthetic only'}]}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root
