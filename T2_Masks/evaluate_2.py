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


def read_excel(filename,sheet):
    df = pd.read_excel(io=file_name, sheet_name=sheet)
    print(df.head(5))  # print first 5 rows of the dataframe
    return(df)

def main():
    header = ("DSC",)
    masking_functions = (get_liver_mask,)
    rows = list()
    subject_ids = list()
    
    disease_file = os.path.abspath("diagnoses.csv")
    df_disease = pd.read_csv(disease_file)
    df_disease.set_index('Key', inplace=True)

    mri_file = os.path.abspath("mri_reports.csv")
    df_mri = pd.read_csv(mri_file)
    df_mri.set_index('Key', inplace=True)

    
    details_file = os.path.abspath("imagedetails.csv")
    df_details = pd.read_csv(details_file)
    df_details.set_index('Key', inplace=True)

    
    for case_folder in glob.glob("prediction_test2_LS200/*"):
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
    print('Index of df:', df.index,'\n \n')
 

    print('Index of mri: ',df_mri.index,'\n \n')
   # print(df_mri.head())


   # Join MRI report
    df_i = pd.concat([df,df_mri],axis=1,join='inner')

    # Join Diagnoses
    df_id = pd.concat([df_i,df_disease],axis=1,join='inner')


    # Join Image Details
    df_idd = pd.concat([df_id,df_details],axis=1,join='inner')

    print(df_idd.index,'\n \n')
    print(df_idd.head())
 
    df_idd.to_csv("./prediction_test2_LS200/Dice_scores_test2.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot_test2.png")
    plt.close()






    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph_test1_train2.png')


if __name__ == "__main__":
    main()
