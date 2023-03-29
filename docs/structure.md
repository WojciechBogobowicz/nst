# Project Structure
The project structure is shown on schema below.  
You can familiarize yourself with project modules by reading short description on the end of each.  
Some of schema modules are bound to their documentation, so you can learn more about them by clicking in it. Every documentation start with brief introduction to module.

----

<span style="font-family:Ubuntu">
**├── README.md**  
**├── [images/](images.md)** &emsp; *stores input and output images*     
**│   ├── contents/** &emsp; *stores content images*              
**│   ├── styles/**  &emsp; *stores style images*               
**│   ├── results/** &emsp; *stores generated images*               
**│   └── trainers/** &emsp; *stores trainers visualizations*            
**├── src/**    &emsp; *contains project source code*  
**│   ├── [utils/](utils.md)**    &emsp; *contains helper functions*  
**│   │   ├── \_\_init\_\_.py**  &emsp; *makes python treat utils as package*  
**│   │   ├── randomizers.py**    &emsp; *contains random trainer init helpers*  
**│   │   └── tf_utils.py**    &emsp; *contains model.py helpers*  
**│   ├── [model.py](model.md)**    &emsp; *NST model source code*  
**│   ├── [selectioner.py](selectioner.md)**  &emsp; *automatically chooses best NST models*    
**│   └── [visualization.py](vizualization.md)**    &emsp; *deals with plots*  
**├── [requirements/](installation.md)** &emsp; *stores configuration files for python env*     
**│   ├── environment.yaml/** &emsp; *defines conda env*                 
**│   ├── pip_requirements.txt/**  &emsp; *defines pip env*                          
**│   └── raw_requirements.txt/**  &emsp; *defines pip env without version specified*     
**├── nst_training.ipynb**    &emsp; *demonstrates simple image generation process*  
**├── manual_training.ipynb**    &emsp; *demonstates harder image generation process*  
**├── mkdocs.yml**    &emsp; *documentation config*  
**└── .gitignore**  &emsp; *defines files ignored by git*  
</span>
