import os


def print_directory_tree(path, prefix = "", exclude_folders = None):
	"""Recursively print a directory tree while excluding specified folders."""
	if exclude_folders is None:
		exclude_folders = []

	# Get a list of all files and directories at the current path
	entries = os.listdir(path)

	# Sort the list to print directories first
	entries = sorted(entries, key = lambda x: (os.path.isfile(os.path.join(path, x)), x.lower()))

	for index, entry in enumerate(entries):
		# Skip excluded folders
		if entry in exclude_folders:
			continue

		# Check if it's the last item in the current level
		is_last = index == len(entries) - 1 or all(e in exclude_folders for e in entries[index + 1:])

		# Use specific characters for tree formatting
		connector = "└── " if is_last else "├── "
		print(f"{prefix}{connector}{entry}")

		# Recursively print subdirectories
		full_path = os.path.join(path, entry)
		if os.path.isdir(full_path):
			new_prefix = f"{prefix}    " if is_last else f"{prefix}│   "
			print_directory_tree(full_path, new_prefix, exclude_folders)


# Run this script
if __name__ == "__main__":
	# Replace '.' with the path to your project if executing from elsewhere
	project_path = "."
	print("Project Directory Structure:")
	print_directory_tree(project_path, exclude_folders = [".venv", "__pycache__", ".git",".idea"])
