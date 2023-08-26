# Objaverse Rendering

## Installation

Following [PyBlend](https://github.com/anyeZHY/PyBlend) to install the dependencies and [Blender](https://www.blender.org/download/) itself.

1. Download Blender from [here](https://www.blender.org/download/).
    
    Once you unzip the file, you will find a folder structured like this:

    ```bash
    ./blender-3.3.1-linux-x64
    ├── blender
    ├── 3.3
    ...
    ```
    where `blender` is the executable file (I will use `{path/to/blender}` or `blender_app` to represent this path in the following) and `3.3` contains the Python environment for Blender.

2. Download `get-pip.py` and install pip for Blender Python.

    ```bash
    $ wget https://bootstrap.pypa.io/get-pip.py

    $ ./blender-3.3.1-linux-x64/3.3/python/bin/python3.10 get-pip.py
    ```

3. Install PyBlend.

    ```bash
    $ ./blender-3.3.1-linux-x64/3.3/python/bin/pip install git+https://github.com/anyeZHY/PyBlend.git
    ```

4. You could install other packages in the same way. E.g.,

    ```bash
    $ ./blender-3.3.1-linux-x64/3.3/python/bin/pip install torch
    ```

## Rendering

Run the following command to render the object with the given UID.

```bash
$ {path/to/blender} -b -P render.py -noaudio --disable-crash-handler -- --uid {ObjaverseUID. E.g., f6e9ec5953854dff94176c36b877c519}
```

## Visualization

```
$ python vis.py --uid {ObjaverseUID. E.g., f6e9ec5953854dff94176c36b877c519}
```