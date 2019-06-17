Cardenity
=========

You can view the notebooks without downloading any dataset or running anything.
Clone repo and run `jupyter notebook` in command line/terminal.

- Run `./download_dataset.sh`
    * This will download the dataset and unzip in `datasets` directory
- Open notebook: `Data Analysis and Data Genration.ipynb`
    * This is where you can check out data analysis steps. Commentaries are in the notebook.
- Open notebook: `Model Training.ipynb`
    * This is where you can check model training procedures. Commentaries are in the notebook.
- Open notebook: `Model Testing.ipynb`
    * This is where you can check model being evaluated on test set and qualitative results on test set.
     Commentaries are in the notebook.


## Ideas

- **Label separation training**: Original labels are further broken down for the model
to treat learning make and model as multi-class multiple prediction problem.
(Details in `Data Analysis and Data Generation` Notebook)

- **Synthetic Data Generation**: The original dataset is quite small and unbalanced. Data
can be balanced by synthetically generating more data from original images.
(Details in `Data Analysis and Data Generation` Notebook)

- **Data Augmentation**: Every car has a character and even seeing A-pillar and B-pillar
will allow experts and enthusiasts to identify the make, model and year of the car
pretty accurately. As a car enthusiast myself, I have tried doing that exercise
to get an idea of human baseline on identifying the vehicle just from parts of
it. Understanding the nature of data very well before actually building a deep learning
model is absolutely necessary and domain knowledge can help a lot as well. Thus,
the images will be cropped, flipped, rotated, saturation, hue, brightness changed.
Random cropping should help a lot.
