# PistonCup

Recommendation app to know every transaction available about your cars.

This app was made just for machine learning development experience. I do not recommend a true use of the app.

This is the first version of the app. It has been developed with supervised machine learning methods like LightGBM. The dataset was found on [HuggingFace]([https://huggingface.co/datasets/cs-uche/car_dealership]), from the user *cs-uche*.

The goal of the app is verify how much can ML algorithms predict prices of cars depending on the parameters introduced. 

## Environment

Python is the main language used in this project. The main libraries used are:

- Pandas, Matplotlib, Seaborn are used to data analysis.
- LightGBM is responsible to generate predictions.
- Flask is the application support of the algorithm.
  
The environment is available on the repository to use it. (Model/ML.yml).
To install it, download the .yml file, open Anaconda Prompt and write:

```
  conda env create -f ML.yml
```

## Authors

- [@alegonzjur](https://www.github.com/alegonzjur)