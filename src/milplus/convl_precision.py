"""Validate the exact original default GradScaler state across every window."""
import math


def validate_scaler_history(use_amp, histories, saved):
    traces = [trace for epoch in histories for trace in epoch]
    if not traces and saved is None:
        return
    if not use_amp:
        if saved != {} or any(trace['skipped'] or trace['scale_before'] != 1. or trace['scale_after'] != 1. for trace in traces):
            raise ValueError('ConVL FP32 reference cannot silently become an AMP route')
        return
    keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
    if (not isinstance(saved, dict) or set(saved) != keys
            or saved['growth_factor'] != 2. or saved['backoff_factor'] != .5 or saved['growth_interval'] != 2000
            or type(saved['_growth_tracker']) is not int or not 0 <= saved['_growth_tracker'] < 2000
            or not isinstance(saved['scale'], (int, float)) or not math.isfinite(saved['scale']) or saved['scale'] <= 0):
        raise ValueError('ConVL original AMP scale/growth/backoff parameters changed')
    scale, consecutive = 65536., 0
    for trace in traces:
        if trace['scale_before'] != scale:
            raise ValueError('ConVL AMP history lost a genuine prior scale or window')
        if trace['skipped']:
            scale *= .5; consecutive = 0
        else:
            consecutive += 1
            if consecutive == 2000:
                scale *= 2.; consecutive = 0
        if trace['scale_after'] != scale:
            raise ValueError('ConVL AMP history changed an actual overflow/growth transition')
    if saved['scale'] != scale or saved['_growth_tracker'] != consecutive:
        raise ValueError('ConVL saved AMP scaler differs from complete actual update history')
