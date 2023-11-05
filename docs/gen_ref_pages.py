"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

rootdir = "src"
for path in sorted(Path(rootdir).rglob("*.py")):
    module_path = path.relative_to(rootdir).with_suffix("")
    doc_path = path.relative_to(rootdir).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        # In this case, the nav link and identifier corresponds to the module
        # name
        parts = parts[:-1]
        nav_link = parts
        identifier = ".".join(parts)
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        # In this case, we inject this documentation as
        # torchdnf/modulename/Tests in the navigation
        identifier = ".".join(parts)
        nav_link = parts[:-1] + ("Tests",)
        doc_path = doc_path.with_name("tests.md")
        full_doc_path = full_doc_path.with_name("tests.md")
    else:
        # That's the normal case, the entry in the navigation
        # and the mkdocs identifier correspond to torchdnf/modulename/submodule
        identifier = ".".join(parts)
        nav_link = parts

    # Add the entry to the navigation
    nav[nav_link] = doc_path.as_posix()

    # Fill in the md file for mkdocs
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}")

# Add the examples
doc_path = "examples/mnist.md"
full_doc_path = Path("reference", doc_path)
nav[("Tutorials", "MNIST")] = doc_path
with mkdocs_gen_files.open(full_doc_path, "w") as fd:
    fd.write("::: examples.mnist\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
