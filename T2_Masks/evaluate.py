import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_liver_mask(data):
    return data > 0

def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def main():
    header = ("DSC",)
    masking_functions = (get_liver_mask,)
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob("prediction_NewTest_3DUnet_isense_overlap16/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("./prediction_NewTest_3DUnet_isense_overlap16/Dice_scores.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot_NewTest_581model.png")
    plt.close()

    if os.path.exists("./training_1GPU.log"):
        training_df = pd.read_csv("./training_1GPU.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('prediction_T1_581model.png')


if __name__ == "__main__":
    main()
