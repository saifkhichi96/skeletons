from differential_skeletons import create

model = create("spinetrack")

# Print model information (joints, hierarchy, offsets)
print("NUM_JOINTS:", model.NUM_JOINTS)
print("NUM_BODIES:", model.NUM_BODIES)
print("Parents:", model.parents)
print("Children:", model.child_body_indices)

# Print model parameters
print("scales:", model.scales.shape)
print("global_orient:", model.global_orient.shape)
print("body_pose:", model.body_pose.shape)
print("transl:", model.transl.shape)
