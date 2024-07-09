from GPT2.utils.common import DirectoryTree
import os
from pathlib import Path
# Assuming you have a folder named 'my_project' in your current directory
root_directory = Path(os.getcwd())
directory_tree = DirectoryTree(root_directory)
directory_tree.write_to_file('tree_output.txt')