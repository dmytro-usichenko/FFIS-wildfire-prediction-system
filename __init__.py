from pathlib import Path

files = [
    "config/__init__.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/training/__init__.py",
    "src/evaluation/__init__.py",
    "src/explainability/__init__.py",
    "src/visualization/__init__.py",
    "src/utils/__init__.py",
    "app/__init__.py",
    "app/pages/__init__.py",
    "app/components/__init__.py",
    "app/state/__init__.py",
]

for file in files:
    Path(file).touch(exist_ok=True)