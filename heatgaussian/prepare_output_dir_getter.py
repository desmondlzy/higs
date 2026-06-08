from pathlib import Path
import shutil
import datetime

def prepare_output_dir_getter(filename: str, name: str = None, on_existed: str = "rename"):
	datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	name = name if name is not None else "unnamed"
	subdir_name = f"{datetime_str}_{name}" if name is None else name

	output_dir = Path(filename).parent / "outputs" / Path(filename).stem / subdir_name
	if output_dir.exists():
		if on_existed == "rename":
			datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			output_dir = output_dir.with_name(f"{output_dir.name}_{datetime_str}")

			if output_dir.exists():  # if still clash, use microseconds
				datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
				output_dir = output_dir.with_name(f"{output_dir.name}_{datetime_str}")

		elif on_existed == "overwrite":
			shutil.rmtree(output_dir)
		else:
			raise ValueError(f"Unknown on_existed option: {on_existed}")
		
	print("🗂️ Writing to output directory:", output_dir)

	output_dir.mkdir(parents=True, exist_ok=True)
	shutil.copy(filename, output_dir)

	def _get_output_dir(subdir=None) -> Path:
		if subdir is None:
			return output_dir

		subdir_path = output_dir / subdir
		subdir_path.mkdir(parents=True, exist_ok=True)
		return subdir_path

	return _get_output_dir
