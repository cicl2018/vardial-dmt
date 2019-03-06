"""
From: Karsper
Borda Algorithm Flow
- Get the number of models used n_model
- Assign a rank for each model according to their accuracy
- Assign a score according to their rank, rank 1 has n_model score, and n_model - 1 onwards

- During the testing phase, classes M or T will be awarded scores with each classifier outcome.
- he class with higher total score will be the actual predicted class.

- Output the final prediction after Borda Fusion.

"""
import operator

if __name__ == '__main__':
    # !Input: Models we used
    m1 = [2, 'character', 0.85]
    m2 = [3, 'character', 0.9]
    m3 = [4, 'character', 0.65]
    m4 = [5, 'character', 0.75]
    m5 = [6, 'character', 0.7]
    models = [m1, m2, m3, m4, m5]

    # !Input: Predicted dialects from each model
    m1_dialect = ['M', 'T', 'M', 'T', 'T']
    m2_dialect = ['T', 'M', 'T', 'M', 'M']  # model with highest accuracy in overall
    m3_dialect = ['T', 'M', 'M', 'M', 'T']
    m4_dialect = ['M', 'T', 'M', 'T', 'T']
    m5_dialect = ['M', 'T', 'M', 'T', 'M']

    at_dialect = ['M', 'T', 'M', 'T', 'T']  # actual dialect
    bo_dialect = [None] * len(at_dialect)   # !Output: Prediction with Borda

    models_dialect = [m1_dialect, m2_dialect, m3_dialect, m4_dialect, m5_dialect]

    # Borda Algorithm
    # relate model accuracy with each model
    models_dict = {}
    for i in range(len(models)):
        models_dict[i+1] = models[i][2]

    # sort models with descending accuracy
    sorted_model = sorted(models_dict.items(), key=operator.itemgetter(1))
    sorted_model = list(reversed(sorted_model))
    best_model = models_dialect[sorted_model[0][0] - 1]  # as a precaution if score_T = score_M

    # prepare an empty score list
    score_list = [None] * len(models)

    # assign scores for each model from model 1 to n
    score = len(models)
    for i in range(len(models)):
        score_list[sorted_model[i][0]-1] = score
        score -= 1

    # Borda Fusion: Fuse the predictions from all models
    for predict in range(len(at_dialect)):  # fuse every prediction from each model in parallel
        score_T = 0  # total score of prediction 'T'
        score_M = 0  # total score of prediction 'M'
        for model in range(len(models_dialect)):
            if models_dialect[model][predict] == 'T':  # add score for T-prediction
                score_T = score_T + score_list[model]
            elif models_dialect[model][predict] == 'M':  # add score for M-prediction
                score_M = score_M + score_list[model]

        if score_T > score_M:
            bo_dialect[predict] = 'T'
        elif score_M > score_T:
            bo_dialect[predict] = 'M'
        else:  # use the prediction from model with highest overall accuracy
            bo_dialect[predict] = best_model[predict]

    # Results
    print("")
    # print("               Borda Score List:", score_list)  # the final Borda score list

    print("            Actual Dialect:", at_dialect)
    print("     Best Model Prediction:", m2_dialect, "  0% Accuracy (Worst Case)")  # simply use the best prediction
    print("   Borda Fusion Prediction:", bo_dialect, "100% Accuracy (Best Case)")  # Borda fusion prediction
    print("")
