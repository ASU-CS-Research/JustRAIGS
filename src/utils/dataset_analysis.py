import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from loguru import logger

# Load the dataset: src/utils/JustRAIGS_Train_labels.csv

# document = pd.read_csv('JustRAIGS_Train_labels.csv', header=0)

with open('JustRAIGS_Train_labels.csv', 'r') as file:
    heading = next(file)

    document = csv.reader(file, delimiter=';')

    positive_agreement = []
    negative_agreement = []
    left_positive_disagreement = []
    left_negative_disagreement = []

    for row in document:
        # row = document.iloc[:,i]
        if row[4] == row[5]:
            if row[4] == 'RG':
                positive_agreement.append(row)
            else:
                negative_agreement.append(row)
        else:
            if row[4] == 'RG':
                left_positive_disagreement.append(row)
            else:
                left_negative_disagreement.append(row)
        # if i == 10:
        #     break

    output = f"There are {len(positive_agreement)} positive agreements, {len(negative_agreement)} negative agreements, " \
             f"{len(left_positive_disagreement)} positive disagreements, and {len(left_negative_disagreement)} negative disagreements."
    print(output)

    left_features = {
        "ANRS": 0,
        "ANRI": 0,
        "RNFLDS": 0,
        "RNFLDI": 0,
        "BCLVS": 0,
        "BCLVI": 0,
        "NVT": 0,
        "DH": 0,
        "LD": 0,
        "LC": 0,
    }

    right_features = {
        "ANRS": 0,
        "ANRI": 0,
        "RNFLDS": 0,
        "RNFLDI": 0,
        "BCLVS": 0,
        "BCLVI": 0,
        "NVT": 0,
        "DH": 0,
        "LD": 0,
        "LC": 0,
    }

    features = ["ANRS", "ANRI", "RNFLDS", "RNFLDI", "BCLVS", "BCLVI", "NVT", "DH", "LD", "LC"]


    # for row in left_positive_disagreement:
    #     print(f"{row[0]}:\t{row[1]}\t{row[17:27]}")

    for row in left_positive_disagreement:
        if '1' in row[17:27]:
            logger.debug(f"{row[0]}:\t{row[1]}\t{row[17:27]}")

    for row in left_positive_disagreement:
        for i in range(10):
            try:
                left_features[features[i]] = left_features[features[i]] + int(row[i + 7])
                right_features[features[i]] = right_features[features[i]] + int(row[i + 17])
            except ValueError:
                continue

    print("\n")
    print("Left Positive Disagreement")
    for feat in features:
        print(f"{feat}:\t\t{left_features[feat]}\t\t{right_features[feat]}")

    left_features = {
        "ANRS": 0,
        "ANRI": 0,
        "RNFLDS": 0,
        "RNFLDI": 0,
        "BCLVS": 0,
        "BCLVI": 0,
        "NVT": 0,
        "DH": 0,
        "LD": 0,
        "LC": 0,
    }

    right_features = {
        "ANRS": 0,
        "ANRI": 0,
        "RNFLDS": 0,
        "RNFLDI": 0,
        "BCLVS": 0,
        "BCLVI": 0,
        "NVT": 0,
        "DH": 0,
        "LD": 0,
        "LC": 0,
    }

    # for row in left_negative_disagreement:
    #     print(f"{row[0]}:\t{row[1]}\t{row[17:27]}")

    for row in left_negative_disagreement:
        if '1' in row[7:16]:
            logger.debug(f"{row[0]}:\t{row[1]}\t{row[7:16]}")

    for row in left_negative_disagreement:
        for i in range(10):
            try:
                left_features[features[i]] = left_features[features[i]] + int(row[i + 7])
                right_features[features[i]] = right_features[features[i]] + int(row[i + 17])
            except ValueError:
                continue

    print("\n")

    print("Left Negative Disagreement")
    for feat in features:
        print(f"{feat}:\t\t{left_features[feat]}\t\t{right_features[feat]}")

    logger.info("the above shows that when their is disagreement, the grader stating that the image is NRG nevr identifies any of the features of Glaucoma while the other does.")


    logger.info("features and graders:")
    for row in document:
        if row[4] == 'NRG' and '1' in row[7:16]:
            logger.debug(f"Grader 1: {row[0]}:\t{row[1]}\t{row[4]}\t{row[7:16]}")
        if row[5] == 'NRG' and '1' in row[17:26]:
            logger.debug(f"Grader 2: {row[0]}:\t{row[1]}\t{row[5]}\t{row[17:26]}")

    logger.info("Grader 3:")
    logger.info("features and graders:")
    for row in document:
        try:
            if row[6] == 'NRG' and '1' in row[27:36]:
                logger.debug(f"{row[0]}:\t{row[1]}\t{row[6]}\t{row[27:36]}")
        except IndexError:
            continue

    row_disagreements = []
    feature_disagreements = {
        "ANRS": 0,
        "ANRI": 0,
        "RNFLDS": 0,
        "RNFLDI": 0,
        "BCLVS": 0,
        "BCLVI": 0,
        "NVT": 0,
        "DH": 0,
        "LD": 0,
        "LC": 0,
    }

    for row in document:
        if row[4] != row[5]:
            row_disagreements.append(row)
            for i in range(10):
                try:
                    add_one = int(row[i + 7]) != int(row[i + 17])
                    feature_disagreements[features[i]] = feature_disagreements[features[i]] + add_one
                except ValueError:
                    continue

    logger.info(f"Row disagreements: {len(row_disagreements)}")
    logger.info(f"Feature disagreements:")
    for feat in features:
        logger.info(f"{feat}:\t\t{feature_disagreements[feat]}")















# compare the things that the graders agree on and the things that they disagree on

# What is the higher level grader likely to agree on?

# are the graders inconsistent with themselves?





