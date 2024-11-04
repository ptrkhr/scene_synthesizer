############
Installation
############

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

The most basic installation of ``scene_synthesizer`` will install as few dependencies as possible:

.. code-block:: bash

    # clone repository
    cd scene_synthesizer/
    pip install -e .

This allows you to load and export most common formats: STL, PLY, OBJ, GLB, GLTF, URDF.

For importing and exporting USD files use the ``recommend`` pip extra:

.. code-block:: bash

    # clone repository
    cd scene_synthesizer/
    pip install -e .[recommend]
    
If you want the full experience, you can try the ``all`` extra which includes all the functionality:

.. code-block:: bash

    # clone repository
    cd scene_synthesizer/
    pip install -e .[all]


Soft Dependencies
=================

The function ``show_graph`` depends on `pygraphviz <http://pygraphviz.github.io/>`_.
If required, install via

.. code-block:: bash

    sudo apt-get install graphviz graphviz-dev
    python -m pip install pygraphviz
