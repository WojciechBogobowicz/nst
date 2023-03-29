# Model
`model.py` is a core file that contains definition of model used in neural style training.  
This model is split into two classes. [`StyleContentExtractor`][src.model.StyleContentExtractor] that implements forward pass of the model and [`NSTImageTrainer`][src.model.NSTImageTrainer] that uses it to perform training.   



::: src.model