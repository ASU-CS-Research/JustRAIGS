import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from loguru import logger


# Load the dataset: src/utils/JustRAIGS_Train_labels.csv

document = pd.read_csv('JustRAIGS_Train_labels.csv', header=0, sep=';')
print(document.head())
# Add together all classifications from the three graders and use the sum as the number of classifications for each image.
classifications = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
for classification in classifications:
    document[classification] = document[f'G1 {classification}'] + document[f'G2 {classification}'] + document[f'G3 {classification}']

# plot the bar chart
plt.bar(classifications, document[classifications].sum())
# plt.legend()
plt.tight_layout()
plt.show()
plt.clf()

classifications = ['RG', 'NRG']
pos_class = document[document['Final Label'] == 'RG']
neg_class = document[document['Final Label'] == 'NRG']
plt.title(f"Binary Class Distribution")
plt.bar(classifications, [pos_class.shape[0], neg_class.shape[0]])
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()
plt.clf()

# with (open('src/utils/JustRAIGS_Train_labels.csv', 'r') as file):
#
#     general_stats = {
#
#     }
#
#     heading = next(file)
#
#     reader = csv.reader(file, delimiter=';')
#
#     positive_agreement = []
#     negative_agreement = []
#     left_positive_disagreement = []
#     left_negative_disagreement = []
#
#     document = []
#
#     for row in reader:
#         # row = document.iloc[:,i]
#
#         document.append(row)
#
#         if row[4] == row[5]:
#             if row[4] == 'RG':
#                 positive_agreement.append(row)
#             else:
#                 negative_agreement.append(row)
#         else:
#             if row[4] == 'RG':
#                 left_positive_disagreement.append(row)
#             else:
#                 left_negative_disagreement.append(row)
#         # if i == 10:
#         #     break
#
#     output = f"There are {len(positive_agreement)} positive agreements, {len(negative_agreement)} negative agreements, " \
#              f"{len(left_positive_disagreement)} positive disagreements, and {len(left_negative_disagreement)} negative disagreements."
#     print(output)
#
#     left_features = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     right_features = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     features = ["ANRS", "ANRI", "RNFLDS", "RNFLDI", "BCLVS", "BCLVI", "NVT", "DH", "LD", "LC"]
#
#
#     # for row in left_positive_disagreement:
#     #     print(f"{row[0]}:\t{row[1]}\t{row[17:27]}")
#
#     for row in left_positive_disagreement:
#         if '1' in row[17:27]:
#             logger.debug(f"{row[0]}:\t{row[1]}\t{row[17:27]}")
#
#     for row in left_positive_disagreement:
#         for i in range(10):
#             try:
#                 left_features[features[i]] = left_features[features[i]] + int(row[i + 7])
#                 right_features[features[i]] = right_features[features[i]] + int(row[i + 17])
#             except ValueError:
#                 continue
#
#     print("\n")
#     print("Left Positive Disagreement")
#     for feat in features:
#         print(f"{feat}:\t\t{left_features[feat]}\t\t{right_features[feat]}")
#
#     left_features = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     right_features = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     # for row in left_negative_disagreement:
#     #     print(f"{row[0]}:\t{row[1]}\t{row[17:27]}")
#
#     for row in left_negative_disagreement:
#         if '1' in row[7:16]:
#             logger.debug(f"{row[0]}:\t{row[1]}\t{row[7:16]}")
#
#     for row in left_negative_disagreement:
#         for i in range(10):
#             try:
#                 left_features[features[i]] = left_features[features[i]] + int(row[i + 7])
#                 right_features[features[i]] = right_features[features[i]] + int(row[i + 17])
#             except ValueError:
#                 continue
#
#     print("\n")
#
#     print("Left Negative Disagreement")
#     for feat in features:
#         print(f"{feat}:\t\t{left_features[feat]}\t\t{right_features[feat]}")
#
#     logger.info("the above shows that when their is disagreement, the grader stating that the image is NRG nevr identifies any of the features of Glaucoma while the other does.")
#
#
#     logger.info("features and graders:")
#     # document = csv.reader(file, delimiter=';')
#     for row in document:
#         if row[4] == 'NRG' and '1' in row[7:16]:
#             logger.debug(f"Grader 1: {row[0]}:\t{row[1]}\t{row[4]}\t{row[7:16]}")
#         if row[5] == 'NRG' and '1' in row[17:26]:
#             logger.debug(f"Grader 2: {row[0]}:\t{row[1]}\t{row[5]}\t{row[17:26]}")
#
#     logger.info("Grader 3:")
#     logger.info("features and graders:")
#     for row in document:
#         try:
#             if row[6] == 'NRG' and '1' in row[27:36]:
#                 logger.debug(f"{row[0]}:\t{row[1]}\t{row[6]}\t{row[27:36]}")
#         except IndexError:
#             continue
#
#     row_disagreements = []
#     feature_disagreements = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     # document = csv.reader(file, delimiter=';')
#     for row in document:
#         if row[4] != row[5]:
#             row_disagreements.append(row)
#             for i in range(10):
#                 try:
#                     add_one = int(row[i + 7]) != int(row[i + 17])
#                     feature_disagreements[features[i]] = feature_disagreements[features[i]] + add_one
#                 except ValueError:
#                     continue
#
#     logger.info(f"Row disagreements: {len(row_disagreements)}")
#     logger.info(f"Feature disagreements:")
#     for feat in features:
#         logger.info(f"{feat}:\t\t{feature_disagreements[feat]}")
#
#
#     # counting each subclass
#     logger.info("Counting each subclass\n")
#
#     left_sub_class_count = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     right_sub_class_count = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     for row in document:
#         for i in range(10):
#             try:
#                 left_sub_class_count[features[i]] = left_sub_class_count[features[i]] + int(row[i + 7])
#                 right_sub_class_count[features[i]] = right_sub_class_count[features[i]] + int(row[i + 17])
#             except ValueError:
#                 continue
#
#     logger.info("Left Subclass Count\n")
#     for feat in features:
#         logger.info(f"{feat}:\t\t{left_sub_class_count[feat]}")
#
#     print("\n")
#
#     logger.info("Right Subclass Count\n")
#     for feat in features:
#         logger.info(f"{feat}:\t\t{right_sub_class_count[feat]}")
#
#     print("\n")
#
#
#
#
#     # which features are most agreed upon?
#
#     agreements = {
#         "ANRS": 0,
#         "ANRI": 0,
#         "RNFLDS": 0,
#         "RNFLDI": 0,
#         "BCLVS": 0,
#         "BCLVI": 0,
#         "NVT": 0,
#         "DH": 0,
#         "LD": 0,
#         "LC": 0,
#     }
#
#     for row in document:
#         if row[4] == row[5] and row[4] == 'RG':
#             for i in range(10):
#                 try:
#                     if row[i + 7] == row[i + 17]:
#                         agreements[features[i]] = agreements[features[i]] + 1
#                 except ValueError:
#                     continue
#
#     logger.info("Agreements\n")
#
#     for feat in features:
#         logger.info(f"{feat}:\t\t{agreements[feat]}")
#
#     print("\n")
#
#     # confusion matrix
#
#     logger.info("Confusion Matrix: The matrix show the times when one feature was identified and another was not identified for each of the 10 features for both graders\n")
#
#     logger.info("Confusion Matrix (left grader)\n")
#
#     left_confusion_matrix = {
#         "ANRS": {},
#         "ANRI": {},
#         "RNFLDS": {},
#         "RNFLDI": {},
#         "BCLVS": {},
#         "BCLVI": {},
#         "NVT": {},
#         "DH": {},
#         "LD": {},
#         "LC": {},
#     }
#
#     for i, key in enumerate(left_confusion_matrix):
#         key_row = {}
#         for j, feat in enumerate(features):
#             key_sum = 0
#             sum = 0
#             for row in document:
#                 if row[7 + i] == '1':
#                     key_sum = key_sum + 1
#                     if row[7 + j] == '1':
#                         sum = sum + 1
#             frequency = float(sum) / float(key_sum)
#             # convert frequency to percentage and round to 2 decimal places
#             frequency = round(frequency * 100, 2)
#             if frequency < 0.01:
#                 frequency = 0
#             key_row[feat] = frequency
#         left_confusion_matrix[key] = key_row
#
#     cm_top_row = "\t\t"
#     for feat in features:
#         cm_top_row = cm_top_row + f"{feat}\t"
#         if len(feat) < 4:
#             cm_top_row = cm_top_row + "\t"
#
#     print(cm_top_row)
#     for key in left_confusion_matrix:
#         row = f"{key}:\t"
#         if len(key) < 3:
#             row = row + "\t"
#         for feat in features:
#             row = row + f"{left_confusion_matrix[key][feat]}\t"
#             if len(str(left_confusion_matrix[key][feat])) < 4:
#                 row = row + "\t"
#         print(row)
#
#     # confusion matrix
#
#     logger.info("Confusion Matrix (right grader)\n")
#
#     right_confusion_matrix = {
#         "ANRS": {},
#         "ANRI": {},
#         "RNFLDS": {},
#         "RNFLDI": {},
#         "BCLVS": {},
#         "BCLVI": {},
#         "NVT": {},
#         "DH": {},
#         "LD": {},
#         "LC": {},
#     }
#
#     for i, key in enumerate(right_confusion_matrix):
#         key_row = {}
#         for j, feat in enumerate(features):
#             key_sum = 0
#             sum = 0
#             for row in document:
#                 if row[17 + i] == '1':
#                     key_sum = key_sum + 1
#                     if row[17 + j] == '1':
#                         sum = sum + 1
#                 else:
#                     _ = 0 # breakpoint
#             frequency = float(sum) / float(key_sum)
#             # convert frequency to percentage and round to 2 decimal places
#             frequency = round(frequency * 100, 2)
#             if frequency < 0.01:
#                 frequency = 0
#             key_row[feat] = frequency
#         right_confusion_matrix[key] = key_row
#
#     cm_top_row = "\t\t"
#     for feat in features:
#         cm_top_row = cm_top_row + f"{feat}\t"
#         if len(feat) < 4:
#             cm_top_row = cm_top_row + "\t"
#
#     print(cm_top_row)
#
#     for key in right_confusion_matrix:
#         row = f"{key}:\t"
#         if len(key) < 3:
#             row = row + "\t"
#         for feat in features:
#             row = row + f"{right_confusion_matrix[key][feat]}\t"
#             if len(str(right_confusion_matrix[key][feat])) < 4:
#                 row = row + "\t"
#         print(row)



# compare the things that the graders agree on and the things that they disagree on

# What is the higher level grader likely to agree on?

# are the graders inconsistent with themselves?





