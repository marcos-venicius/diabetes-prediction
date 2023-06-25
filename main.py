#!/usr/bin/env python3

import pandas as pd
import joblib
import settings

df = pd.read_csv(settings.DATASET_PATH)

class GetData:
    def __init__(self):
        pass

    def __yes_or_not(self, title):
        data = {'yes': 1, 'not': 0}

        r = input(f'{title} [yes, not]: ')

        if r not in data:
            print(f'[!] invalid {title}')
            return self.__yes_or_not(title)

        return data[r]

    def __get_from_map(self, data, title):
        options = ', '.join([i for i in data.keys()])

        r = input(f"{title} [{options}]: ")

        if r not in data:
            print(f'[!] invalid {title}')
            return self.__get_from_map(data, title)

        return data[r]

    def gender(self):
        return self.__get_from_map(settings.GENDER_MAPPED, 'Gender')

    def smoking_history(self):
        return self.__get_from_map(settings.SMOKING_HISTORY_MAPPED, 'Smoking history')

    def age(self):
        r = input("Age: ")

        if not r.isnumeric() or int(r) < 0:
            print("[!] invalid Age")
            return self.age()

        return int(r)

    def hypertension(self):
        return self.__yes_or_not('Hypertension')

    def heart_disease(self):
        return self.__yes_or_not('Heart disease')

    def bmi(self):
        r = input("BMI: ")

        if not r.replace('.', '').isnumeric():
            print("[!] invalid BMI")
            return self.bmi()

        return float(r)

    def hba1c_level(self):
        r = input("HBA1C level: ")

        if not r.replace('.', '').isnumeric():
            print("[!] invalid HBA1C level")
            return self.hba1c_level()

        return float(r)

    def blood_glucose_level(self):
        r = input("Blood glucose level: ")

        if not r.isnumeric():
            print("[!] invalid Blood glucose level")
            return self.blood_glucose_level()

        return int(r)



get_data = GetData()

gender = get_data.gender()
age = get_data.age()
hypertension = get_data.hypertension()
heart_disease = get_data.heart_disease()
smoking_history = get_data.smoking_history()
bmi = get_data.bmi()
hba1c_level = get_data.hba1c_level()
blood_glucose_level = get_data.blood_glucose_level()

tree = joblib.load(settings.WEIGHTS_PATH)

prediction = tree.predict([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]])

if prediction[0] == 1:
    print("Result: possible positive")
else:
    print("Result: possible negative")

