#!/usr/bin/env python3
"""
Simple test to verify the NaN replacement logic works correctly.
"""

import math

def _replace_nan_with_default(data_dict, default_value=-1.0):
    """
    Replace NaN values in a dictionary with a default value.
    
    This is necessary because DataJoint cannot properly format queries 
    with NaN values, causing errors during probe insertion/validation.
    
    Args:
        data_dict: Dictionary that may contain NaN values
        default_value: Value to replace NaN with (default: -1.0)
    
    Returns:
        Dictionary with NaN values replaced
    """
    if not isinstance(data_dict, dict):
        return data_dict
        
    result = data_dict.copy()
    for key, value in result.items():
        if isinstance(value, float) and math.isnan(value):
            result[key] = default_value
    
    return result

# Test the function
test_data = {
    'probe_id': 'test',
    'contact_size': float('nan'),
    'rel_x': float('nan'),
    'rel_y': 5.0,
    'rel_z': float('nan')
}

print('Original:', test_data)
result = _replace_nan_with_default(test_data)
print('After replacement:', result)

# Verify no NaN values remain
for key, value in result.items():
    if isinstance(value, float) and math.isnan(value):
        print(f'ERROR: NaN still present in {key}')
    else:
        print(f'✓ {key}: {value}')

# Test the issue case
print('\n' + '='*50)
print('Testing the exact case from the issue:')
null_val = float('nan')
key = {
    'probe_id': 'nTrode32_probe description', 
    'probe_shank': 0,
    'contact_size': null_val, 
    'probe_electrode': 194,
    'rel_x': null_val,
    'rel_y': null_val, 
    'rel_z': null_val
}
print('Original key:', key)
fixed_key = _replace_nan_with_default(key)
print('Fixed key:', fixed_key)

# Verify no NaN values remain
has_nan = False
for k, v in fixed_key.items():
    if isinstance(v, float) and math.isnan(v):
        has_nan = True
        print(f'ERROR: NaN still present in {k}')

if not has_nan:
    print('✓ All NaN values successfully replaced!')
    print('✓ This key should now work with DataJoint queries!')