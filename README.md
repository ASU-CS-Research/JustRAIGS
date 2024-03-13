# JustRAIGS
ASU: Justified Referral in AI Glaucoma Screening challenge.

## Setup Instructions:
1. Login to the lambda machine (via ssh key is preferred). To set this up:
   1. Alias the lambda machine in your `~/.ssh/config` file:
      ```bash
       Host lambda
             HostName 152.10.212.186
             User <your-username>
             Port 22
      ```
   2. Issue a key exchange with the lambda machine:
      ```bash
        ssh-copy-id <your-username>@lambda
      ```
   3. You should now be able to login to the lambda machine with the following command:
      ```bash
      ssh <your-username>@lambda
      ```
2. Create a new Python virtual environment using [`venv`](https://docs.python.org/3/library/venv.html). Ensure you use
the command below to do so, as this will leverage the [lambda-stack](https://lambdalabs.com/lambda-stack-deep-learning-software) by default:
   ```bash
   python3 -m venv just-raigs --system-site-packages
   ```
3. If you are not in `bash` shell, switch to it:
    ```bash
    bash
    ```
4. Activate the virtual environment:
   ```bash
   source just-raigs/bin/activate
   ```
5. Note that the [lambda-stack](https://lambdalabs.com/lambda-stack-deep-learning-software) will have installed most of
the deep learning packages you require for you:
    ```bash
    pip list
    ```
6. I have created an environment for us to use, which is stored in the `requirements.txt` file. You can install all the
required packages with the following command:
   ```bash
   pip install -r requirements.txt
   ```
7. Any other packages you wish to install can be installed with [`pip`](https://pip.pypa.io/en/stable/):
   ```bash
   pip install some-package
   ```

## Resource Monitoring and Courtesy:
1. The lambda machine is powerful, but it is not a GPU cluster. There are two `NVIDIA GeForce RTX 3090 GPUs` and `126 GB`
of RAM that we must all share. 
   1. Please note that [AppMAIS](https://appmais.cs.appstate.edu/) work must take precedence over the challenge, as
   this machine was purchased with [AppMAIS](https://appmais.cs.appstate.edu/) grant funds.
2. Please monitor resource consumption and utilization with the following commands:
   1. Current GPU usage: `watch -d -n 0.5 nvidia-smi`
   2. Memory usage: `htop`
3. If one of the GPUs is not in use, you can use it. If both are in use, please wait until one is free. Communicate with
your team members on our `JustRAIGS` Google Chat Space to coordinate experimentation and resource utilization.
4. Ensure you check [WaB](https://wandb.ai/appmais/JustRAIGS) to see if someone is already running an experiment that
would include the parameters you were going to run. 


## Data Location:
1. Data is stored in the `/usr/local/data/JustRAIGS` directory.
2. The raw compressed files are stored in `/usr/local/data/JustRAIGS/compressed`.
3. The extracted uncompressed files (unmodified) are stored in: `/usr/local/data/JustRAIGS/raw`.
   1. Files in this directory are partitioned by the original dataset splits provided by the challenge organizers. For
      example, the training data is stored in `/usr/local/data/JustRAIGS/raw/train/0` corresponds to the compressed 
      file: `JustRAIGS_Train_0.zip` provided directly from the challenge Zenodo website.
4. I have provided a utility method :meth:`src.utils.datasets.load_datasets` which will load the training datasets from 
   the disk, perform preprocessing, rescaling, normalization, and convert the result into TensorFlow Datasets for use 
   downstream.

## Introduction to Weights and Biases (WaB):
1. [Weights and Biases (WaB)](https://appmais.cs.appstate.edu/) is a tool that allows us to track and visualize our 
   experiments. It is a great tool for collaboration, and is flexible enough to be customized how you see fit. 
   1. This means you have the freedom to log arbitrary metrics, artifacts, models, and more.
   2. You can also integrate WaB with other deep learning frameworks and libraries. For instance, you could use 
      [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) with WaB instead of the native WaB 
      hyperparameter tuning framework. 
2. I have created a WaB project for us to use, which is located at: [WaB: JustRAIGS](https://wandb.ai/appmais/JustRAIGS).
3. I have also provided comments in [Restructured Text (RST)](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) 
   format in the codebase to help you understand how to integrate WaB with your code, and which documentation to 
   reference when you get stuck.
4. There is some high level terminology you should know which will allow you to utilize WaB effectively:
   1. [`Organization`](https://docs.wandb.ai/guides/app/features/organizations): This is a collection of `Projects`. It 
      is a way to organize separate distinct `Projects` within a particular organization/research group.
      1. `Project`: This is the highest level of organization in WaB. It is a collection of [`Experiments`](https://docs.wandb.ai/guides/track).
   2. An [`Experiment`](https://docs.wandb.ai/guides/track) is a collection of 
      [`Runs`](https://docs.wandb.ai/guides/runs), [`Metrics`](https://docs.wandb.ai/guides/track/log/log-summary) and 
      [`Artifacts`](https://docs.wandb.ai/guides/track/log/media). 
      1. A [`Run`](https://docs.wandb.ai/guides/runs): Is a single unit of computation logged by WaB. Consider a 
         [`Run`](https://docs.wandb.ai/guides/runs) an atomic element of the whole project. A new 
         [`Run`](https://docs.wandb.ai/guides/runs) should be initialized if you change a hyperparameter, use a 
         different model, etc. 
      2. Within a [`Run`](https://docs.wandb.ai/guides/runs), [`Metrics`](https://docs.wandb.ai/guides/track/log/log-summary) 
         can be tracked across the training and validation datasets and visualized in the WaB dashboard. You could track
         the `accuracy`, `loss`, `precision`, `recall` and more as [`Metrics`](https://docs.wandb.ai/guides/track/log/log-summary).
      3. Within a [`Run`](https://docs.wandb.ai/guides/runs), [`Artifacts`](https://docs.wandb.ai/guides/track/log/media) 
         can also be logged. For example, you could log the model weights, the model architecture, the training dataset, 
         the training logs, a matplotlib image, etc.
         1. For more on logging artifacts, see the [WaB Documentation on Artifacts](https://docs.wandb.ai/guides/artifacts).
   3. WaB uses [`Reports`](https://docs.wandb.ai/guides/reports) to organize [`Runs`](https://docs.wandb.ai/guides/runs), 
      embed visualizations, describe findings, and share content with collaborators.
   4. WaB leverages [`Sweeps`](https://docs.wandb.ai/guides/sweeps) to automate hyperparameter searching and perform 
      experiment tracking. [`Sweeps`](https://docs.wandb.ai/guides/sweeps) allow you to leverage popular hyperparameter
      search methods such as Bayesian search, grid search, random search, hyperband search, and more. 
      [`Sweeps`](https://docs.wandb.ai/guides/sweeps) require a 
      [`Sweep Configuration`](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) to specify the overall 
      hyperparameter search space and the method of search. 
      1. Behind the scenes, the [WaB Sweep Controller](https://docs.wandb.ai/guides/sweeps/walkthrough#initialize-the-sweep) 
         manages the execution of the [`Sweep`](https://docs.wandb.ai/guides/sweeps). You interact with the 
         [WaB Sweep Controller](https://docs.wandb.ai/guides/sweeps/walkthrough#initialize-the-sweep) via the 
         [`wandb.agent` API](https://docs.wandb.ai/guides/sweeps/walkthrough#start-the-sweep).
   5. WaB [`Sweeps`](https://docs.wandb.ai/guides/sweeps) generate unique `Trials` based on the [`Sweep Configuration`](https://docs.wandb.ai/guides/sweeps/walkthrough#initialize-the-sweep) provided.
      Each `Trial` is a unique subset of the overall hyperparameter search space specified by the [`Sweep Configuration`](https://docs.wandb.ai/guides/sweeps/walkthrough#initialize-the-sweep).
      In the way I leverage WaB, each `Trial` will have its own unique [`Run`](https://docs.wandb.ai/guides/runs), and 
      therefore its own [`Metrics`](https://docs.wandb.ai/guides/track/log/log-summary) and 
      [`Artifacts`](https://docs.wandb.ai/guides/track/log/media).
   6. In Keras a [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) is a class that performs fitting 
      (weight optimization) to a particular datasets. The model is in charge of minimizing a particular loss function, 
      and is capable of making predictions on a particular dataset (i.e. performing inference). This class also is in 
      charge of: resource management for the training process, the logging of metrics, the logging of losses, and the 
      logging of artifacts to WaB.
   7. A `Hypermodel` is a class that is in charge of instantiating and managing `Models` for the hyperparameter tuning 
       process. The `HyperModel` class is instantiated just once per-`Sweep` and is responsible for creating a new 
       `Model` for each `Trial` in the `Sweep`.

## A Walkthrough of the Code Provided:
1. In the `docs` directory, you will find the source files needed to build [Sphinx](https://www.sphinx-doc.org/en/master/)
   documentation. [Sphinx](https://www.sphinx-doc.org/en/master/) is a static website generator that parses Python 
   docstrings into HTML documentation. This is the tool leveraged by the [official Python documentation](https://docs.python.org/3/) 
   so it is worth your while to be somewhat familiar with it. Sphinx operates on docstrings written in [Restructured Text
   (RST)](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) form. Restructured Text is a superset
   of [Markdown](https://www.markdownguide.org/). Since RST can be ugly to look at, I write docstrings in the 
   [Google Documentation Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), and use 
   the [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension to
   parse it into RST, which Sphinx then utilizes to generate pretty looking HTML documentation. You will most likely not
   need to know how this works. Just know that if you write good docstrings in the Google Style 
   [(use this example for reference)](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)
   , Sphinx will be able to generate readable documentation for you (almost) automagically.
   1. If you use PyCharm (which you should) for Python development, then PyCharm can build the Sphinx documentation for 
      you. Additionally, you can configure PyCharm to lint your docstrings in Google Style. Ask me if you want to know
      how to do this.
2. In the `src` directory, you will find the following relevant subdirectories:
   1. `sweepers`: This is the main entry point for the program. The :mod:`src.sweepers.sweeper` module is the file you 
      should modify to either change the hyperparameters that are experimented with, or to change the method of the 
      hyperparameter search itself (i.e. random search, grid search, hyperband, etc.). Note that if you do change the 
      hyperparameters, you will also need to change the `Hypermodel` itself to be able to handle the new hyperparameters.
   2. `hypermodels`: Contains an example :class:`hypermodels.hypermodels.WaBHyperModel` class which is instantiated by
      the WaB agent just once for a particular `Sweep`. This class is responsible for creating a new `Model` for each
      `Trial` (i.e. unique subset of hyperparameters). Specifically, the 
      :meth:`hypermodels.hypermodels.WaBHyperModel.construct_model_run_trial` method is invoked once per `Trial` and is
      in charge of creating a new `Model` for the `Trial` and fitting the `Model` for the `Trial`. You will need to 
      modify this method if you modify the hyperparameters in the sweep configuration.
   3. `models`: Contains an example :class:`models.models.WaBModel` class which is instantiated by the `Hypermodel`
      once for every `Trial`. This class is separate from the hypermodel as the `hypermodel` could theoretically 
      instantiate separate `Model` subclasses for each `Trial`. Additionally, the :class:`models.models.WaBModel` class
      provides an example of how to perform custom model serialization and deserialization in TensorFlow 2.0. If you 
      wish to use a non-sequential model, or a model that requires custom serialization/deserialization, this class 
      will serve as a useful reference.
   4. `metrics`: This file is used to house custom metrics that are not available by default in the Keras API. Note that
      the use of custom metrics will result in a custom model, which means you will have to modify the serialization and
      deserialization methods in the :class:`src.models.models.WaBModel` class.
   5. `utils`: This directory houses utility functions which are leveraged by the various classes above. For instance,
      the :meth:`utils.datasets.load_datasets` method will load the training dataset from the disk, perform 
      preprocessing, rescaling and normalization, and convert the result into TensorFlow Datasets for use downstream. 
   6. `layers`: This directory houses the :mod:`src.layers.custom` module which provides an example of how to create 
      custom layers in TensorFlow 2.0. This is not used in the current codebase, but is provided as a reference in case
      you wish to use custom layers in your model. Note that using a custom layer will result in a custom module, which
      will require you to modify the serialization an deserialization methods in the :class:`src.models.WaBModel` class.
3. Less-relevant subdirectories:
   1. `tuners`: This directory houses the :mod:`src.tuners.wab_kt_tuner` module which provides an example of how to 
      integrate WaB with Kerastuner. This is not used in the current codebase, but is provided as a reference in case
      you wish to leverage KerasTuner for hyperparameter tuning directly, instead of WaB. This class uses KerasTuner
      as a driver, but still integrates with WaB for experiment tracking, versioning, and artifact retention. 