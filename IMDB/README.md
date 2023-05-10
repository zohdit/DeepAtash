# DeepAtash-IMDB #

Focused test generator for DL systems

## General Information ##
This folder contains the application of the DeepAtash approach to the movie sentiment analysis problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.6.


### Configuring Ubuntu ###
Pull an Ubuntu Docker image, run and configure it by typing in the terminal:

``` 
docker pull ubuntu:bionic
docker run -it --rm ubuntu:bionic
apt update && apt-get update
apt-get install -y software-properties-common
```

### Installing git ###
Use the following command to install git:

``` 
apt install -y git
```

### Copy the project into the docker container ###

To copy DeepAtash-IMDB inside the docker container, open another console and run:

``` 
cd <DEEP_ATASH_HOME>
docker cp DeepAtash-IMDB/ <DOCKER_ID>:/
```

Where `<DEEP_ATASH_HOME>` is the location in which you downloaded the artifact and `<DOCKER_ID>` is the ID of the ubuntu docker image just started.

You can find the id of the docker image using the following command:

```
docker ps -a

CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS          PORTS     NAMES
13e590d65e60   ubuntu:bionic   "/bin/bash"   2 minutes ago   Up 2 minutes             recursing_bhabha
```

### Installing Python 3.6 ###
Install Python 3.6:

``` 
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.6
```

And check if it is correctly installed, by typing the following command:

``` 
python3 -V

Python 3.6.9
```

Check that the version of python matches `3.6.*`.

### Installing pip ###

Use the following commands to install pip and upgrade it to the latest version:

``` 
apt install -y python3-pip
python3 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3 -m pip --version

pip 21.1.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)
```
### Creating a Python virtual environment ###

Install the `venv` module in the docker container:

``` 
apt install -y python3-venv
```

Create the python virtual environment:

```
cd /DeepAtash-IMDB
python3 -m venv .venv
```

Activate the python virtual environment and updated `pip` again (venv comes with an old version of the tool):

```
. .venv/bin/activate
pip install --upgrade pip
```


### Installing Dependencies ###

This tool has other dependencies, including `tensorflow` and `deap`, that can be installed via `pip`:

```
cd /DeepAtash-IMDB
pip install -r requirements.txt
``` 

## Usage ##
### Input ###

* A trained model in h5 format. The default one is in the folder `models`;
* `config.py` containing the configuration of the tool selected by the user.

### Run the Tool ###

To run each of the search approaches use the following command:

```
python ga_method.py
```
or
```
python nsga2_method.py
```

### Output ###

When the run is finished, the tool produces the following outputs in the `logs` folder:

* folders containing the generated inputs (in image format).


## Troubleshooting ##

* if pip cannot install the correct version of `opencv-python` check whether you upgraded pip correctly after you activate the virtual environment `.venv`

* If tensorflow cannot be installed successfully, try to upgrade the pip version. Tensorflow cannot be installed by old versions of pip. We recommend the pip version 20.1.1.

* If the import of cairo, potrace or other modules fails, check that the correct version is installed. The correct version is reported in the file requirements.txt. The version of a module can be checked with the following command:

```
pip3 show modulename | grep Version
```
    
To fix the problem and install a specific version, use the following command:
    
```
pip3 install 'modulename==moduleversion' --force-reinstall
```
