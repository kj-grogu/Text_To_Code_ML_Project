https://www.kaggle.com/code/rhtsingh/text-to-code-generation-with-tensorflow-mbpp


# Create the virtual env
python -m venv .env


# Activate the virtual environment
source .env/bin/activate

# Deactivate the virtual environment
source .env/bin/deactivate



# install datasets package in the virtual environment
pip install datasets

# install transformers package in the virtual environment
pip install transformers


pip install 'transformers[torch]'

pip install 'transformers[tf-cpu]'

IF getting error on above use cmak + pkg config 
(https://huggingface.co/docs/transformers/installation)
brew install cmake
brew install pkg-config



  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [5 lines of output]
      Package sentencepiece was not found in the pkg-config search path.
      Perhaps you should add the directory containing `sentencepiece.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'sentencepiece' found
      Failed to find sentencepiece pkgconfig
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.



Install tensorflow

https://www.tensorflow.org/install/pip#macos_1

pip install tensorflow


Verify TensorFlow installation

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"