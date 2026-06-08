
def transform_local_to_hemicube(v_local):
	source_to_camera_dir_hemicube = v_local.clone()
	source_to_camera_dir_hemicube[..., 0] = -source_to_camera_dir_hemicube[..., 0]
	
	return source_to_camera_dir_hemicube
