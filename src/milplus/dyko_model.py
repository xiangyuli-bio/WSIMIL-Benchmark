"""The full actual DyKo class, initialized by the original pretrained TITAN."""
import importlib.metadata

import torch
from torch import nn

from .dyko_assets import OriginalTITANAssets
from .dyko_source import author_classes, source_assets


def build_original(inputs, backend, *, require_source_faiss_version=True):
    import faiss
    if require_source_faiss_version and str(faiss.__version__) != '1.8.0':
        raise ValueError('Use the explicit original FAISS1.8 runtime; source requirements specify faiss-cpu1.8.0.post1')
    native = inputs.titan if hasattr(inputs,'titan') else OriginalTITANAssets(inputs['titan_assets'])
    cohort = inputs.cohort if hasattr(inputs,'cohort') else inputs['cohort']
    asset = source_assets(cohort);evidence = []

    def load_titan():
        model = native.load()
        evidence.append(model._milplus_original_titan_evidence)
        return model

    original = author_classes(backend=backend,titan_loader=load_titan)
    model = original.DyKo(input_size=768,text_prompt_path=str(asset['prompts']),prototype_number=16,
                         num_clusters=10,num_concepts=10,num_heads=8,n_classes=asset['classes'])
    if len(evidence) != 1 or not evidence[0]['all_original_model_tensors_exact']:
        raise ValueError('The complete original DyKo constructor did not use the actual pretrained TITAN')
    for name in ['Path_Adapter','Text_Adapter']:
        layers = getattr(model,name).fc
        if (not isinstance(layers[0],nn.Linear) or (layers[0].in_features,layers[0].out_features) != (768,192)
                or (layers[2].in_features,layers[2].out_features) != (192,768)
                or layers[0].bias is not None or layers[2].bias is not None):
            raise ValueError('Full native DyKo adapter dimensions changed')
    if (model.num_clusters != 10 or model.num_concepts != 10
            or model.prompt_learner.ctx.shape[0] != 16
            or any(getattr(model,name).embed_dim != 768 or getattr(model,name).num_heads != 8
                   for name in ['cross_attention0','cross_attention1'])
            or (model.classifier.in_features,model.classifier.out_features) != (768,asset['classes'])):
        raise ValueError('Full original DyKo prompts/heads/clusters/concept counts changed')
    # The source does not freeze the selected contextual transformer, positional
    # embedding, normalization or projection. Preserve their actual updates.
    text = dict(model.text_encoder.named_parameters())
    if not text or any(not p.requires_grad for p in text.values()):
        raise ValueError('Actual original TITAN contextual text parameters must remain trainable')
    prompt_length = model.prompt_learner.token_prefix.shape[1]+16+model.prompt_learner.token_suffix.shape[1]
    tokens = model.prompt_learner.tokenized_prompts
    if (tokens.ndim != 2 or len(tokens) != asset['classes']
            or prompt_length != model.text_encoder.positional_embedding.shape[0]-1
            or bool((tokens.argmax(-1) >= prompt_length).any())):
        raise ValueError('Original DyKo contextual EOT pooling lies outside its actual token sequence')
    model._milplus_dyko_evidence = {'original_TITAN':evidence[0],
        'actual_model_parameters':sum(p.numel() for p in model.parameters()),
        'actual_trainable_text_parameters':sum(p.numel() for p in text.values()),
        'actual_text_parameter_names':list(text),'text_tower_frozen':False,
        'faiss_backend':backend,'faiss_version':str(faiss.__version__),
        'source_dependency_conflict':'source requests gpu=True but published requirements pin faiss-cpu1.8.0.post1',
        'concept_count':len(asset['bank']),'feature_dim':768,'prompts':16,'heads_each':8,
        'cluster_count':10,'selected_concepts_per_cluster':10,
        'source_class_count':asset['classes'],'native_prompt_sequence_length':prompt_length,
        'source_concepts_sha256':asset['concepts_sha256'],'source_prompts_sha256':asset['prompts_sha256'],
        'pretrained_factory_allocations_preserved':True,'full_method_validated':False}
    return model,original.backend
