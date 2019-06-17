Cardenity
=========

- Run ./download_dataset.sh
- Open notebook: `Data Analysis and Data Genration.ipynb`
- Open notebook: `Model Training.ipynb`


Identity make and model of cars from images.

## Ideas before implementation
 
- **Label separation training**: Original labels are further broken down for the model 
to treat learning make and model as multi-task learning problem

- **Data augmentation**: Every car has a character and even seeing A-pillar and B-pillar
will allow experts and enthusiasts to identify the make, model and year of the car
pretty accurately. As a car enthusiast myself, I have tried doing that exercise 
to get an idea of human baseline on identifying the vehicle just from parts of
it. Understanding the nature of data very well before actually building a deep learning 
model is absolutely necessary and domain knowledge can help a lot as well. Thus,
the images will be cropped, flipped, rotated, saturation, hue, brightness changed.
Random cropping should help a lot.
 