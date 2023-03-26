# Installation
## Without virtual envs:
*This approach isn't recommended due to potential dependency conflicts that may occur during installation process, but if it will work in your default environment it would be the fastest way.*
```console
pip install -r ./requirements/raw_requrements.txt
```
## Conda installation:
```console
conda create --name nst_env --file /requirements/conda_requirements.txt.
```


## Pip installation:
*If you want to create virtual env by your own in different way (for example using ide), then to it and go to step four.*
1. Install library for virtual environments if you don't have it yet.
```console
pip install virtualenv
```
2. Create new virtual environment.
```console
virtualenv nst_env
```
3. Activate it:
```console
./test_env/Scripts/Activate.ps1
```
4. Install requirements from file:
```console
pip install -r ./requirements/pip_requirements.txt
```

