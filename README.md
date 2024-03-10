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
3. Activate the virtual environment:
   ```bash
   source just-raigs/bin/activate
   ```
4. Note that the [lambda-stack](https://lambdalabs.com/lambda-stack-deep-learning-software) will have installed most of
the deep learning packages you require for you:
    ```bash
    pip list
    ```
5. I have created an environment for us to use, which is stored in the `requirements.txt` file. You can install all the
required packages with the following command:
   ```bash
   pip install -r requirements.txt
   ```
6. Any other packages you wish to install can be installed with [`pip`](https://pip.pypa.io/en/stable/):
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

