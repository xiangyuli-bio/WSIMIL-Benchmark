"""Explicit, versioned two-level labels; never infer clinical hierarchies from class IDs."""


class HierarchyLabels:
    def __init__(self, bundle):
        value = bundle.config['hierarchy']
        if bundle.task_type not in {'classification', 'ordinal'}:
            raise ValueError('hierarchy requires a classification or ordinal task')
        required = {'coarse_classes', 'fine_to_coarse', 'label_provenance'}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f'hierarchy must contain exactly {sorted(required)}')
        self.fine = list(bundle.classes)
        self.coarse = value['coarse_classes']
        if (not isinstance(self.coarse, list) or
                any(not isinstance(x, str) or not x.strip() for x in self.coarse) or
                len(set(self.coarse)) != len(self.coarse) or not 2 <= len(self.coarse) <= len(self.fine)):
            raise ValueError('coarse_classes must contain 2..fine_count unique nonempty names')
        mapping = value['fine_to_coarse']
        if (not isinstance(mapping, dict) or set(mapping) != set(self.fine) or
                any(not isinstance(x, str) for x in mapping.values()) or
                set(mapping.values()) != set(self.coarse)):
            raise ValueError('fine_to_coarse must map every fine class exactly once and cover every coarse class')
        provenance = value['label_provenance']
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError('hierarchy label_provenance is required')
        self.mapping = tuple(self.coarse.index(mapping[name]) for name in self.fine)
        if 'coarse_label' in bundle.rows and not bundle.rows.coarse_label.eq(bundle.rows.label.map(mapping)).all():
            raise ValueError('coarse_label disagrees with registered fine_to_coarse mapping')
        self.provenance = provenance
        self.evaluation_unit = bundle.config.get('evaluation_unit', 'slide')

    def protocol(self):
        return {'fine_classes': self.fine, 'coarse_classes': self.coarse,
                'fine_to_coarse': dict(zip(self.fine, (self.coarse[i] for i in self.mapping))),
                'label_provenance': self.provenance, 'evaluation_unit': self.evaluation_unit}
