# Optimium Runtime Python Example App

This example describes how to run classification model (mobilenet v3) on Linux with Optimium.

> This example assumes your target device is x64 (AMD64, x86_64) architecture CPU.

## How to build

1. Create virtual environment.
    ``` bash
    python3 -m venv .venv
    . .venv/bin/activate
    ```

1. Install requisites.
    ``` bash
    pip3 install -f $OPTIMIUM_SDK_ROOT/runtime/python -r requirements.txt
    ```

1. Copy the model into your working directory. The model file (`mobilenetv3.tflite`) is in the `Models` folder of the repository.

1. Optimize the model using Optimium. Please refer the document to get detailed description.
    ``` bash
    # Create a template
    $OPTIMIUM_SDK_ROOT/run_optimium.sh --working_dir workdir --create_template

    # edit your user_arguments.json: "YOUR_MODEL.tflite" to "mobilenetv3.tflite"

    # Optimize the model
    $OPTIMIUM_SDK_ROOT/run_optimium.sh --working_dir workdir
    ```

1. Copy optimized model folder (NOT contents of the model folder).
    ``` bash
    cp -r workdir/.../mobilenetv3 .
    ```

1. Open `wsgi.py` with your favorite editor and edit model name.
    ![Edit location of the model name](assets/python_edit_model_name.png)

1. Run the app.
    ``` bash
    FLASK_DEBUG=True python3 -m flask run
    ```

1. Open `http://127.0.0.1:5000` with your browser and upload some picture. Classification results shows on left of the picture.
    ![Classification result on brower](assets/sample_python.png)
