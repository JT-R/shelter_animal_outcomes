source("utilities.R")

labeled_data <- LoadData("train.csv")
testing <- LoadData("test.csv")
sample_submission <- LoadData("sample_submission.csv")

processed_datasets <- PreprocessDatasets(labeled_data, testing)
labeled_data <- processed_datasets[['labeled_data']]
testing <- processed_datasets[['testing']]

# 
# model <- gbm(OutcomeType ~ Sex + Has_Name + DaysUponOutcome + AnimalType + Color + Fixed_Breed,
#              data = labeled_data,
#              n.trees = 150)
# summary(model)
# predictions <- data.table(predicted_class = predict(model, newdata = labeled_data,
#      n.trees = 150, type = "response"))

######
# TO PONIZEJ BYLO NAJLEPSZE W DRUGIM SUBMISSION:
######
# model <- multinom(OutcomeType ~ Sex + Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed + TimeOfTheDay + Season + Year + Month + Day, data = labeled_data)
# FitUnknownLevelsToModel(dataset = testing, model = model)
# 
# predictions <- data.table(predicted_class = predict(model, newdata = testing, type = "probs"))
# setnames(predictions, colnames(predictions), gsub(pattern = "predicted_class.", replacement = "", x = colnames(predictions), fixed=TRUE))
# setcolorder(predictions, neworder = order(colnames(predictions)))
# predictions <- data.table(ID = testing$ID, predictions)
# write.csv(x = predictions, file = "predictions.csv", row.names=FALSE, quote=FALSE)

#####
model <- multinom(OutcomeType ~ Has_Name + DaysUponOutcome + Fixed_Breed + TimeOfTheDay +
                    Season + Year + IsMix + IsDomestic + Sex*IsSterilized + HairLength +
                    NumberOfColors + FirstColor + AnimalType*AgeCategory + IsWeekend + YearPart, data = labeled_data)
FitUnknownLevelsToModel(dataset = testing, model = model)

predictions <- data.table(predicted_class = predict(model, newdata = testing, type = "probs"))
setnames(predictions, colnames(predictions), gsub(pattern = "predicted_class.", replacement = "", x = colnames(predictions), fixed=TRUE))
setcolorder(predictions, neworder = order(colnames(predictions)))
predictions <- data.table(ID = testing$ID, predictions)
write.csv(x = predictions, file = "predictions.csv", row.names=FALSE, quote=FALSE)


model_formula <- as.formula("~Sex + Has_Name + DaysUponOutcome +
                                           AnimalType + Fixed_Breed + TimeOfTheDay + Season +
                                           Year + Month + Day + IsDomestic + IsMix")

model_formula <- as.formula("~Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed + TimeOfTheDay +
                              Season + Year + Month + Day + IsMix + IsDomestic + Sex + IsSterilized + HairLength +
                              NumberOfColors + FirstColor")

xgb_labeled <- xgb.DMatrix(model.matrix(model_formula, data = labeled_data),
                            label=as.numeric(as.factor(labeled_data$OutcomeType))-1, missing=NA)

xgb_testing <- xgb.DMatrix(model.matrix(model_formula, data = testing), missing=NA)

model <- xgboost(data = xgb_labeled,
                 num_class = 5,
                 nrounds = 90,
                 eta = 0.1,
                 nthreads = 4,
                 max.depth = 7,
                 objective = "multi:softprob",
                 eval_metric = "mlogloss")

predictions <- predict(model, xgb_testing)
predictions <- data.table(t(matrix(predictions, nrow = 5, ncol = nrow(testing))))
setnames(predictions, colnames(predictions), c("Euthanasia", "Adoption", "Died", "Return_to_owner", "Transfer"))
setcolorder(predictions, neworder = order(colnames(predictions)))
predictions <- data.table(ID = testing$ID, predictions)

write.csv(x = predictions, file = "predictions.csv", row.names=FALSE, quote=FALSE)







validation_datasets <- CreateValidationDatasets(labeled_set, validation)
training <- validation_datasets[['training']]
validation <- validation_datasets[['validation']]

xgb_training <- xgb.DMatrix(model.matrix(model_formula, data = training),
                         label=as.numeric(as.factor(training$OutcomeType))-1, missing=NA)

xgb_validation <- xgb.DMatrix(model.matrix(model_formula, data = validation),
                            label=as.numeric(as.factor(validation$OutcomeType))-1, missing=NA)

model <- xgboost(data = xgb_training,
                 num_class = 5,
                 nrounds = 90,
                 eta = 0.1,
                 nthreads = 4,
                 max.depth = 7,
                 objective = "multi:softprob",
                 eval_metric = "mlogloss")

predictions <- predict(model, xgb_validation)
predictions <- data.table(t(matrix(predictions, nrow = 5, ncol = nrow(validation))))
setnames(predictions, colnames(predictions), c("Euthanasia", "Adoption", "Died", "Return_to_owner", "Transfer"))
setcolorder(predictions, neworder = order(colnames(predictions)))

MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(predictions))




validation_datasets <- CreateValidationDatasets(labeled_set, validation)
training <- validation_datasets[['training']]
validation <- validation_datasets[['validation']]

model <- multinom(OutcomeType ~ Sex + Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed, data = training)

model <- multinom(OutcomeType ~ Has_Name + DaysUponOutcome + Fixed_Breed + TimeOfTheDay +
                    Season + Year + IsMix + IsDomestic + Sex*IsSterilized + HairLength +
                    NumberOfColors + FirstColor + AnimalType*AgeCategory + IsWeekend + YearPart,
                  data = training, MaxNWts = 10000)
FitUnknownLevelsToModel(dataset = validation, model = model)

predictions <- data.table(predicted_class = predict(model, newdata = validation, type = "probs"))
setnames(predictions, colnames(predictions), gsub(pattern = "predicted_class.", replacement = "", x = colnames(predictions), fixed=TRUE))
setcolorder(predictions, neworder = order(colnames(predictions)))
#predictions <- data.table(ID = validation$AnimalID, predictions)

actual_classes <- data.table(model.matrix(~ OutcomeType -1, data = validation))
setnames(actual_classes, colnames(actual_classes), gsub(pattern = "OutcomeType", replacement = "", x = colnames(actual_classes), fixed=TRUE))
setcolorder(actual_classes, neworder = order(colnames(actual_classes)))

MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(predictions))
#
MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(actual_classes))
