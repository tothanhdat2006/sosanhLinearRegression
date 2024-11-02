import models
import train
import evaluate
import predict


# print(" ---------------------- Training ----------------------")
# import time
# st = time.time()
# train.training_LinearRegression()
# ed = time.time()
# print(">>> Training Linear Regression took ", ed-st, " seconds")
#
# # ------------ Coeffients and intercepts ------------------
# import config
# import dill
# def load_saved_model(final_model_name):
#     with open(config.PATH_model + "/" + final_model_name + ".pkl", "rb") as file:
#         return dill.load(file)
#
# lin_reg = load_saved_model("LinearRegression_model")
# print("Coefficents: ", lin_reg[1].coef_, lin_reg[1].intercept_)
# # ----------------------------------------------------------
#
# st = time.time()
# train.training_RandomForest()
# ed = time.time()
# print(">>> Training Random Forest took ", ed-st, " seconds")
#
# st = time.time()
# train.training_DNN()
# ed = time.time()
# print(">>> Training DNN took ", ed-st, " seconds")


print(" ---------------------- Evaluation ----------------------")
evaluate.eval_LinearRegression()
evaluate.eval_RandomForest()
evaluate.eval_DNN()


# print(" ---------------------- Prediction ----------------------")
# st = time.time()
# predict.predict_LinearRegression()
# ed = time.time()
# print(">>> Predict using Linear Regression took ", ed-st, " seconds")
#
# st = time.time()
# predict.predict_RandomForest()
# ed = time.time()
# print(">>> Predict using Random Forest took ", ed-st, " seconds")
#
# st = time.time()
# predict.predict_DNN()
# ed = time.time()
# print(">>> Predict using DNN took ", ed-st, " seconds")