############
Installation
############

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

There are two ways to install ``scene_synthesizer``: either directly installing from a PyPI server or by cloning the code repository.

Via pip
=======

The most basic installation of ``scene_synthesizer`` will install as few dependencies as possible:

.. code-block:: bash

    pip install scene-synthesizer

This allows you to load and export most common formats: STL, PLY, OBJ, GLB, GLTF, URDF.

For importing and exporting USD files use the ``recommend`` pip extra:

.. code-block:: bash

    pip install scene-synthesizer[recommend]

If you want the full experience, you can try the ``all`` extra which includes all the functionality:

.. code-block:: bash
    
    pip install scene-synthesizer[all]

Via git
=======

Alternatively, if you want to make changes to ``scene_synthesizer``, clone the repository:

.. code-block:: bash

    # create and activate conda env or venv
    git clone https://github.com/NVlabs/scene_synthesizer.git
    cd scene_synthesizer/

    # Again, there are the same three variations you can install:
    # .
    # .[recommend]
    # .[all]
    pip install -e .


Soft Dependencies
=================

The function ``show_graph`` depends on `pygraphviz <http://pygraphviz.github.io/>`_.
If required, install via

.. code-block:: bash

    sudo apt-get install graphviz graphviz-dev
    python -m pip install pygraphviz
