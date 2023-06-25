# Diabetes prediction

This code uses a [kaggle dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

# Disclaimer

I'm not a specialist!
This code is to apply some concepts that i learned on college about data science

**Please, don't understand this results as a diagnosis!**

# How to run?

First of all you need to install dependencies:

```shell
pip install -r ./requirements.txt
```

Then train the model with the dataset

```shell
./train.py
```

Now you can make the prediction based on new data

```shell
./main.py
```

You will be asked about some questions like: gender, age, hypertension, ...

After the program ask you about this questions you will get the answer like:

```
Result: possible positive
```

or 

```
Result: possible negative
```
