# Sphinx makefile with api docs generation
# (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
# (2) make.py generate Python api docs, one ".rst" file per class / function
# (3) make.py calls the actual `sphinx-build`

# Standard Library
import argparse
import importlib
import inspect
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _create_or_clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print("Removed directory %s" % dir_path)
    os.makedirs(dir_path)
    print("Created directory %s" % dir_path)


class PyAPIDocsBuilder:
    """
    Generate Python API *.rst files, per (sub) module, per class, per function.
    The file name is the full module name.
    ...
    """

    def __init__(self, output_dir="python_api", input_dir="python_api_in"):
        """
        input_dir: The input dir for custom rst files that override the
                   generated files.
        """
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.module_names = PyAPIDocsBuilder._get_documented_module_names()
        print("Generating *.rst Python API docs in directory: %s" % self.output_dir)

    def generate_rst(self):
        _create_or_clear_dir(self.output_dir)

        for module_name in self.module_names:
            try:
                module = self._try_import_module(module_name)
                self._generate_module_class_function_docs(module_name, module)
            except:
                print("[Warning] Module {} cannot be imported.".format(module_name))

    @staticmethod
    def _get_documented_module_names():
        """Reads the modules of the python api from the index.rst"""
        module_names = []
        with open("documented_modules.txt", "r") as f:
            for line in f:
                print(line, end="")
                m = re.match("^(scene_synthesizer\..*)\s*$", line)
                if m:
                    module_names.append(m.group(1))
        print("Documented modules:")
        for module_name in module_names:
            print("-", module_name)
        return module_names

    def _try_import_module(self, full_module_name):
        """Returns the module object for the given module path"""
        # SRL
        import scene_synthesizer  # make sure the root module is loaded

        try:
            # Try to import directly. This will work for pure python submodules
            module = importlib.import_module(full_module_name)
            return module
        except ImportError:
            # Traverse the module hierarchy of the root module.
            # This code path is necessary for modules for which we manually
            # define a specific module path (e.g. the modules defined with
            # pybind).
            current_module = scene_synthesizer
            for sub_module_name in full_module_name.split(".")[1:]:
                current_module = getattr(current_module, sub_module_name)
            return current_module

    def _generate_function_doc(self, full_module_name, function_name, output_path):
        out_string = ""
        out_string += "%s.%s" % (full_module_name, function_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name
        out_string += "\n\n" + ".. autofunction:: %s" % function_name
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    def _generate_class_doc(self, full_module_name, class_name, output_path):
        out_string = ""
        out_string += "%s.%s" % (full_module_name, class_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name
        out_string += "\n\n" + ".. autoclass:: %s" % class_name
        out_string += "\n    :members:"
        out_string += "\n    :undoc-members:"
        out_string += "\n    :inherited-members:"
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    def _generate_module_doc(
        self, full_module_name, class_names, function_names, sub_module_names, sub_module_doc_path
    ):
        class_names = sorted(class_names)
        function_names = sorted(function_names)
        out_string = ""
        out_string += full_module_name
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name

        if len(class_names) > 0:
            out_string += "\n\n**Classes**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for class_name in class_names:
                out_string += "\n    " + "%s" % (class_name,)
            out_string += "\n"

        if len(function_names) > 0:
            out_string += "\n\n**Functions**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for function_name in function_names:
                out_string += "\n    " + "%s" % (function_name,)
            out_string += "\n"

        if len(sub_module_names) > 0:
            out_string += "\n\n**Modules**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for sub_module_name in sub_module_names:
                out_string += "\n    " + "%s" % (sub_module_name,)
            out_string += "\n"

        obj_names = class_names + function_names + sub_module_names
        if len(obj_names) > 0:
            out_string += "\n\n.. toctree::"
            out_string += "\n    :hidden:"
            out_string += "\n"
            for obj_name in obj_names:
                out_string += "\n    %s <%s.%s>" % (
                    obj_name,
                    full_module_name,
                    obj_name,
                )
            out_string += "\n"

        with open(sub_module_doc_path, "w") as f:
            f.write(out_string)

    def _generate_module_class_function_docs(self, full_module_name, module):
        print("Generating docs for submodule: %s" % full_module_name)

        # Class docs
        class_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.isclass(obj[1])
            and not obj[0].startswith("_")
            and obj[1].__module__ == module.__name__
        ]
        for class_name in class_names:
            file_name = "%s.%s.rst" % (full_module_name, class_name)
            output_path = os.path.join(self.output_dir, file_name)
            input_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(input_path):
                shutil.copyfile(input_path, output_path)
                continue
            self._generate_class_doc(full_module_name, class_name, output_path)

        # Function docs
        function_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.isroutine(obj[1]) and not obj[0].startswith("_")
        ]
        for function_name in function_names:
            file_name = "%s.%s.rst" % (full_module_name, function_name)
            output_path = os.path.join(self.output_dir, file_name)
            input_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(input_path):
                shutil.copyfile(input_path, output_path)
                continue
            self._generate_function_doc(full_module_name, function_name, output_path)

        # Submodule docs, only supports 2-level nesting
        sub_module_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.ismodule(obj[1]) and not obj[0].startswith("_")
        ]
        documented_sub_module_names = [
            sub_module_name
            for sub_module_name in sub_module_names
            if "%s.%s" % (full_module_name, sub_module_name) in self.module_names
        ]

        # Path
        sub_module_doc_path = os.path.join(self.output_dir, full_module_name + ".rst")
        input_path = os.path.join(self.input_dir, full_module_name + ".rst")
        if os.path.isfile(input_path):
            shutil.copyfile(input_path, sub_module_doc_path)
            return

        self._generate_module_doc(
            full_module_name,
            class_names,
            function_names,
            documented_sub_module_names,
            sub_module_doc_path,
        )


class PyExampleDocsBuilder:
    """
    Generate Python examples *.rst files.
    """

    def __init__(self, input_dir, pwd, output_dir="python_example"):
        self.output_dir = Path(str(output_dir))
        self.input_dir = Path(str(input_dir))
        self.prefixes = [
            ("image", "Image"),
            ("kd_tree", "KD Tree"),
            ("octree", "Octree"),
            ("point_cloud", "Point Cloud"),
            ("ray_casting", "Ray Casting"),
            ("rgbd", "RGBD Image"),
            ("triangle_mesh", "Triangle Mesh"),
            ("voxel_grid", "Voxel Grid"),
        ]

        sys.path.append(os.path.join(pwd, "..", "python", "tools"))

        # from cli import _get_all_examples_dict
        # self.get_all_examples_dict = _get_all_examples_dict
        self.get_all_examples_dict = lambda: {}
        print("Generating *.rst Python example docs in directory: %s" % self.output_dir)

    def _get_examples_dict(self):
        examples_dict = self.get_all_examples_dict()
        # categories_to_remove = ["benchmark", "reconstruction_system", "t_reconstruction_system"]
        categories_to_remove = []
        for cat in categories_to_remove:
            examples_dict.pop(cat)
        return examples_dict

    def _get_prefix(self, example_name):
        for prefix, sub_category in self.prefixes:
            if example_name.startswith(prefix):
                return prefix
        raise Exception("No prefix found for geometry examples")

    @staticmethod
    def _generate_index(title, output_path):
        os.makedirs(output_path)
        out_string = f"{title}\n{'-' * len(title)}\n\n"
        with open(output_path / "index.rst", "w") as f:
            f.write(out_string)

    @staticmethod
    def _add_example_to_docs(example, output_path):
        shutil.copy(example, output_path)
        out_string = (
            f"{example.stem}.py"
            "\n```````````````````````````````````````\n"
            f"\n.. literalinclude:: {example.stem}.py"
            "\n   :language: python"
            "\n   :linenos:"
            "\n   :lineno-start: 27"
            "\n   :lines: 27-"
            "\n\n\n"
        )

        with open(output_path / "index.rst", "a") as f:
            f.write(out_string)

    def generate_rst(self):
        _create_or_clear_dir(self.output_dir)
        examples_dict = self._get_examples_dict()

        # categories = [cat for cat in self.input_dir.iterdir() if cat.is_dir()]
        categories = []

        for cat in categories:
            if cat.stem in examples_dict.keys():
                out_dir = self.output_dir / cat.stem
                if cat.stem == "geometry":
                    self._generate_index(cat.stem.capitalize(), out_dir)
                    with open(out_dir / "index.rst", "a") as f:
                        f.write(f".. toctree::\n    :maxdepth: 2\n\n")
                        for prefix, sub_cat in self.prefixes:
                            f.write(f"    {prefix}/index\n")

                    for prefix, sub_category in self.prefixes:
                        self._generate_index(sub_category, out_dir / prefix)
                    examples = sorted(Path(cat).glob("*.py"))
                    for ex in examples:
                        if ex.stem in examples_dict[cat.stem]:
                            prefix = self._get_prefix(ex.stem)
                            sub_category_path = out_dir / prefix
                            self._add_example_to_docs(ex, sub_category_path)
                else:
                    if cat.stem == "io":
                        self._generate_index("IO", out_dir)
                    else:
                        self._generate_index(cat.stem.capitalize(), out_dir)

                    examples = sorted(Path(cat).glob("*.py"))
                    for ex in examples:
                        if ex.stem in examples_dict[cat.stem]:
                            shutil.copy(ex, out_dir)
                            self._add_example_to_docs(ex, out_dir)


class SphinxDocsBuilder:
    """
    SphinxDocsBuilder calls Python api and examples docs generation and then
    calls sphinx-build:

    (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
    (2) Calls PyAPIDocsBuilder to generate Python api docs rst files
    (3) Calls `sphinx-build` with the user argument
    """

    def __init__(self, current_file_dir, html_output_dir, parallel):
        self.current_file_dir = current_file_dir
        self.html_output_dir = html_output_dir
        self.parallel = parallel

    def run(self):
        """
        Call Sphinx command with hard-coded "html" target
        """
        build_dir = os.path.join(self.html_output_dir, "html")
        nproc = multiprocessing.cpu_count() if self.parallel else 1
        print(f"Building docs with {nproc} processes")

        cmd = [
            "sphinx-build",
            "-j",
            str(nproc),
            "-b",
            "html",
            ".",
            build_dir,
        ]

        sphinx_env = os.environ.copy()

        print('Calling: "%s"' % " ".join(cmd))
        print('Env: "%s"' % sphinx_env)
        subprocess.check_call(cmd, env=sphinx_env, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    """
    # Clean existing notebooks in docs/tutorial, execute notebooks if the
    # notebook does not have outputs, and build docs for Python and C++.
    $ python make_docs.py --clean_notebooks --execute_notebooks=auto --sphinx --doxygen

    # Build docs for Python (--sphinx) and C++ (--doxygen).
    $ python make_docs.py --execute_notebooks=auto --sphinx --doxygen

    # Build docs for release (version number will be used instead of git hash).
    $ python make_docs.py --is_release --sphinx --doxygen
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--py_api_rst",
        default="always",
        choices=("always", "never"),
        help="Build Python API documentation in reST format.",
    )
    parser.add_argument(
        "--py_example_rst",
        default="always",
        choices=("always", "never"),
        help="Build Python example documentation in reST format.",
    )
    parser.add_argument(
        "--sphinx",
        action="store_true",
        default=False,
        help="Build Sphinx for main docs and Python API docs.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable parallel Sphinx build.",
    )
    parser.add_argument(
        "--outputdir",
        default="_out",
        help="Output directory, relative to script location.",
    )

    args = parser.parse_args()

    pwd = os.path.dirname(os.path.realpath(__file__))

    # Clear output dir if new docs are to be built
    html_output_dir = os.path.join(pwd, args.outputdir)
    _create_or_clear_dir(html_output_dir)

    # Python API reST docs
    if not args.py_api_rst == "never":
        print("Building Python API reST")
        pd = PyAPIDocsBuilder()
        pd.generate_rst()

    # Python example reST docs
    py_example_input_dir = os.path.join(pwd, "..", "examples", "python")
    if not args.py_example_rst == "never":
        print("Building Python example reST")
        pe = PyExampleDocsBuilder(input_dir=py_example_input_dir, pwd=pwd)
        pe.generate_rst()

    # Sphinx is hard-coded to build with the "html" option
    # To customize build, run sphinx-build manually
    if args.sphinx:
        print("Building Sphinx docs")
        sdb = SphinxDocsBuilder(pwd, html_output_dir, args.parallel)
        sdb.run()
    else:
        print("Sphinx build disabled, use --sphinx to enable")
