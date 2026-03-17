"""
Marker name mapping dictionary for converting from expected format (with '_study' suffix)
to actual format (without '_study' suffix, lowercase).

This mapping is used to rename markers in TRC files to match the expected format
used by the gait_analysis class.

Expected format: markers end with '_study' (e.g., 'r_calc_study', 'r.ASIS_study')
Actual format: markers without '_study' suffix, lowercase (e.g., 'r_calc', 'r_ASIS')
"""

MARKER_NAME_MAPPING = {
    # Pelvis markers
    'r.ASIS_study': 'r_ASIS',
    'L.ASIS_study': 'l_ASIS',
    'r.PSIS_study': 'r_PSIS',
    'L.PSIS_study': 'l_PSIS',
    
    # Right leg markers
    'r_knee_study': 'r_knee',
    'r_mknee_study': 'r_mknee',
    'r_ankle_study': 'r_ankle',
    'r_mankle_study': 'r_mankle',
    'r_toe_study': 'r_toe',
    'r_5meta_study': 'r_5meta',
    'r_calc_study': 'r_calc',
    
    # Left leg markers
    'L_knee_study': 'l_knee',
    'L_mknee_study': 'l_mknee',
    'L_ankle_study': 'l_ankle',
    'L_mankle_study': 'l_mankle',
    'L_toe_study': 'l_toe',
    'L_calc_study': 'l_calc',
    'L_5meta_study': 'l_5meta',
    
    # Shoulder markers
    'r_shoulder_study': 'r_shoulder',
    'L_shoulder_study': 'l_shoulder',
    
    # Spine markers
    'C7_study': 'C7',
    
    # Thigh markers (may not exist in all files)
    # 'r_thigh1_study': 'r_thigh1',  # Check if exists in actual file
    # 'r_thigh2_study': 'r_thigh2',  # Check if exists in actual file
    # 'r_thigh3_study': 'r_thigh3',  # Check if exists in actual file
    # 'L_thigh1_study': 'l_thigh1',  # Check if exists in actual file
    # 'L_thigh2_study': 'l_thigh2',  # Check if exists in actual file
    # 'L_thigh3_study': 'l_thigh3',  # Check if exists in actual file
    
    # Shank markers (may not exist in all files)
    # 'r_sh1_study': 'r_sh1',  # Check if exists in actual file
    # 'r_sh2_study': 'r_sh2',  # Check if exists in actual file
    # 'r_sh3_study': 'r_sh3',  # Check if exists in actual file
    # 'L_sh1_study': 'l_sh1',  # Check if exists in actual file
    # 'L_sh2_study': 'l_sh2',  # Check if exists in actual file
    # 'L_sh3_study': 'l_sh3',  # Check if exists in actual file
    
    # Hip joint centers
    'RHJC_study': 'RHJC',  # Check if exists in actual file
    'LHJC_study': 'LHJC',  # Check if exists in actual file
    
    # Elbow markers
    # 'r_lelbow_study': 'r_elbow',  # Approximate mapping
    'r_melbow_study': 'r_melbow',
    # 'L_lelbow_study': 'l_elbow',  # Approximate mapping
    'L_melbow_study': 'l_melbow',
    
    # Wrist markers
    # 'r_lwrist_study': 'r_wrist_radius',  # Approximate mapping
    # 'r_mwrist_study': 'r_wrist_ulna',  # Approximate mapping
    # 'L_lwrist_study': 'l_wrist_radius',  # Approximate mapping
    # 'L_mwrist_study': 'l_wrist_ulna',  # Approximate mapping
}

# Reverse mapping (actual -> expected) for renaming markers in TRC files
REVERSE_MARKER_NAME_MAPPING = {v: k for k, v in MARKER_NAME_MAPPING.items()}

# if __name__ == '__main__':
#     """
#     Print the mapping dictionary in a format suitable for use in cloud functions.
#     """
#     import json
    
#     print("Marker name mapping (expected -> actual):")
#     print("=" * 60)
#     for expected, actual in sorted(MARKER_NAME_MAPPING.items()):
#         print(f"  '{expected}' -> '{actual}'")
    
#     print("\n\nReverse mapping (actual -> expected) for renaming:")
#     print("=" * 60)
#     for actual, expected in sorted(REVERSE_MARKER_NAME_MAPPING.items()):
#         print(f"  '{actual}' -> '{expected}'")
    
#     print("\n\nJSON format (for cloud function):")
#     print("=" * 60)
#     print(json.dumps(REVERSE_MARKER_NAME_MAPPING, indent=2))

