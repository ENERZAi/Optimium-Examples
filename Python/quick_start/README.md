# Optimium Runtime Python Example

This example describes how to run face detection model in Python with Optimium.

> This example assumes your device runs on ARM64(AArch64, armv8-a) architecture CPU.

## How to build

1. Copy the model into your working directory. The model file (`face_detection_short_range.tflite`) is in the `Models` folder of the repository.
2. Optimize the model using Optimium.
    ``` bash
    # Create a template
    $OPTIMIUM_SDK_ROOT/run_optimium.sh --working_dir $YOUR_WORKING_DIR --create_template

    # edit your user_arguments.json: "YOUR_MODEL.tflite" to "face_detection_short_range.tflite"

    # Optimize the model
    $OPTIMIUM_SDK_ROOT/run_optimium.sh --working_dir $YOUR_WORKING_DIR
    ```
3. Copy optimized model folder (NOT contents of the model folder) into current folder.
    ``` bash
    cp -rf $YOUR_WORKING_DIR/outputs/{device_name}-{num_thread}-{opt_log_key}/{out_dirname}/ ./optimium_model_output
    ```
4. Run python example
    ``` python
    python3 face_detection_example.py
    ```