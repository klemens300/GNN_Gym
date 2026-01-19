# fix_syntax.py
from pathlib import Path

file = Path("dataset.py")
content = file.read_text()

# Fix the broken line
content = content.replace(
    'final_file = structure_folder / "final_relaxed.cif\n',
    'final_file = structure_folder / "final_relaxed.cif"\n'
)

file.write_text(content)
print("? Fixed syntax error in dataset.py")