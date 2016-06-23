source("utilities.R")

labeled_data <- LoadData("train.csv")
testing <- LoadData("test.csv")
sample_submission <- LoadData("sample_submission.csv")

processed_datasets <- PreprocessDatasets(labeled_data, testing)
labeled_data <- processed_datasets[['labeled_data']]
testing <- processed_datasets[['testing']]

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
final_multinomial_predictions <- data.table(ID = testing$ID, predictions)
write.csv(x = final_multinomial_predictions, file = "mult_predictions.csv", row.names=FALSE, quote=FALSE)


model_formula <- as.formula("~Sex + Has_Name + DaysUponOutcome +
                                           AnimalType + Fixed_Breed + TimeOfTheDay + Season +
                                           Year + Month + Day + IsDomestic + IsMix -1")

model_formula <- as.formula("~ Has_Name + DaysUponOutcome + Fixed_Breed + TimeOfTheDay +
                    Season + Year + IsMix + IsDomestic + Sex + IsSterilized + HairLength +
                    NumberOfColors + FirstColor + AnimalType + AgeCategory + IsWeekend + YearPart -1")

# submission 11:
# predictors <- c("Has_Name","DaysUponOutcome","Fixed_Breed","TimeOfTheDay","Season","Year","IsMix","IsDomestic","Sex",
#                 "IsSterilized","HairLength","NumberOfColors","FirstColor","AnimalType","AgeCategory","IsWeekend","YearPart")
predictors <- c("Has_Name","DaysUponOutcome","Breed","TimeOfTheDay","Season","Day","Year","IsMix","IsDomestic","Sex",
                "IsSterilized","HairLength","NumberOfColors","FirstColor","AnimalType","AgeCategory","IsWeekend","YearPart")

xgb_outcome <- as.numeric(as.factor(labeled_data$OutcomeType))-1

xgb_labeled <- xgb.DMatrix(data.matrix(labeled_data[,predictors, with=FALSE]), label = xgb_outcome)               # the training set
xgb_testing <- xgb.DMatrix(data.matrix(testing[,predictors, with=FALSE]))

model <- xgb.train(data = xgb_labeled,
                   label = xgb_outcome,
                   num_class = 5,
                   nrounds = 160,
                   eta = 0.05,
                   nthreads = 4,
                   #max.depth = 6,
                   watchlist = list(train = xgb_labeled),
                   objective = "multi:softprob",
                   eval_metric = "mlogloss"
                   )

predictions <- predict(model, xgb_testing)
predictions <- data.table(t(matrix(predictions, nrow = 5, ncol = nrow(testing))))
#setnames(predictions, colnames(predictions), c("Euthanasia", "Adoption", "Died", "Return_to_owner", "Transfer"))
#setnames(predictions, colnames(predictions), c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'))
setnames(predictions, colnames(predictions), c('Euthanasia', 'Adoption', 'Died', 'Return_to_owner', 'Transfer'))
setcolorder(predictions, neworder = order(colnames(predictions)))
final_xgb_pred <- predictions
final_xgb_pred <- data.table(ID = testing$ID, final_xgb_pred)

write.csv(x = final_xgb_pred, file = "xgb_predictions.csv", row.names=FALSE, quote=FALSE)
hist(final_xgb_pred$Adoption)
hist(final_xgb_pred$Died)
hist(final_xgb_pred$Euthanasia)
hist(final_xgb_pred$Return_to_owner)
hist(final_xgb_pred$Transfer)


validation_datasets <- CreateValidationDatasets(labeled_set, validation)
training <- validation_datasets[['training']]
validation <- validation_datasets[['validation']]
actual_classes <- data.table(model.matrix(~ OutcomeType -1, data = validation))

xgb_sample_outcome <- as.numeric(as.factor(training$OutcomeType))-1
xgb_validation_outcome <- as.numeric(as.factor(validation$OutcomeType))-1

# predictors <- c("Has_Name","DaysUponOutcome","Fixed_Breed","TimeOfTheDay","Season","Year","IsMix","IsDomestic","Sex",
#                 "IsSterilized","HairLength","NumberOfColors","FirstColor","AnimalType","AgeCategory","IsWeekend","YearPart")

# predictors <- c("Has_Name","DaysUponOutcome","Fixed_Breed","TimeOfTheDay","Month","Day","Year","IsMix","IsDomestic","Sex",
#                 "IsSterilized","HairLength","NumberOfColors","FirstColor","SecondColor","AnimalType","AgeCategory","IsWeekend","YearPart")

predictors <- c("Has_Name","DaysUponOutcome","Fixed_Breed","TimeOfTheDay","Month","Day","Year","IsMix","IsDomestic","Sex",
                "IsSterilized","HairLength","NumberOfColors","FirstColor","SecondColor","AnimalType","AgeCategory","IsWeekend","YearPart")
predictors <- c(predictors, colnames(training)[grepl("stype_", colnames(training))])

predictors <- c("Has_Name","DaysUponOutcome","Fixed_Breed","TimeOfTheDay","Year","Month","Day","IsMix","IsDomestic","Sex",
                "IsSterilized","FirstColor","SecondColor","AnimalType","AgeCategory")
predictors <- c(predictors, colnames(training)[grepl("stype_", colnames(training))])

predictors <- c("Has_Name","DaysUponOutcome","Breed","Month","Day","Year","IsMix","IsDomestic","Sex",
                "IsSterilized","HairLength","FirstColor","AnimalType")

xgb_training <- xgb.DMatrix(data.matrix(training[,predictors, with=FALSE]), label = xgb_sample_outcome)               # the training set
xgb_validation <- xgb.DMatrix(data.matrix(validation[,predictors, with=FALSE]), label = xgb_validation_outcome)

xgb_model <- xgb.train(data = xgb_training,
                       label = xgb_sample_outcome,
                       num_class = 5,
                       nrounds = 80,
                       eta = 0.1,
                       min_child_weight = 0.1,
                       nthreads = 4,
                       max.depth = 7,
                       watchlist = list(train = xgb_training, eval = xgb_validation),
                       objective = "multi:softprob",
                       eval_metric = "mlogloss"
                       )

predictions <- predict(xgb_model, xgb_validation)
predictions <- data.table(t(matrix(predictions, nrow = 5, ncol = nrow(validation))))
# setnames(predictions, colnames(predictions), c("Euthanasia", "Adoption", "Died", "Return_to_owner", "Transfer"))
setnames(predictions, colnames(predictions), c('Euthanasia', 'Adoption', 'Died', 'Return_to_owner', 'Transfer'))
#setcolorder(predictions, neworder = order(colnames(predictions)))
MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(predictions))
xgb_pred <- predictions
#actual_classes <- data.table(model.matrix(~ OutcomeType -1, data = validation))

xgb_imp <- xgb.importance(feature_names = predictors,
                          model = xgb_model)
xgb.plot.importance(xgb_imp)

xgb_classes <- apply(xgb_pred, 2, function(x) ifelse(x>median(x), 1, 0))

xgb_tables <- vector(mode = "list", 5)
xgb_tables[[1]] <- table(xgb_classes[,"Adoption"], validation$Adoption)
xgb_tables[[2]] <- table(xgb_classes[,"Died"], validation$Died)
xgb_tables[[3]] <- table(xgb_classes[,"Euthanasia"], validation$Euthanasia)
xgb_tables[[4]] <- table(xgb_classes[,"Return_to_owner"], validation$Return_to_owner)
xgb_tables[[5]] <- table(xgb_classes[,"Transfer"], validation$Transfer)
xgb_tables
xgb_accuracy <- sapply(xgb_tables, function(x) sum(diag(x))/sum(x))
xgb_accuracy



# 
# validation_datasets <- CreateValidationDatasets(labeled_set, validation)
# training <- validation_datasets[['training']]
# validation <- validation_datasets[['validation']]

# mult_model <- multinom(OutcomeType ~ Sex + Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed, data = training)

mult_model <- multinom(OutcomeType ~ Has_Name + DaysUponOutcome + Fixed_Breed + TimeOfTheDay +
                    Season + Year + IsMix + IsDomestic + Sex*IsSterilized + HairLength +
                    NumberOfColors + FirstColor + AnimalType*AgeCategory + IsWeekend + YearPart,
                  data = training, MaxNWts = 10000)
FitUnknownLevelsToModel(dataset = validation, model = mult_model)

predictions <- data.table(predicted_class = predict(mult_model, newdata = validation, type = "probs"))
setnames(predictions, colnames(predictions), gsub(pattern = "predicted_class.", replacement = "", x = colnames(predictions), fixed=TRUE))
setcolorder(predictions, neworder = order(colnames(predictions)))
mult_pred <- predictions
#predictions <- data.table(ID = validation$AnimalID, predictions)


setnames(actual_classes, colnames(actual_classes), gsub(pattern = "OutcomeType", replacement = "", x = colnames(actual_classes), fixed=TRUE))
setcolorder(actual_classes, neworder = order(colnames(actual_classes)))

MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(predictions))
#
MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(actual_classes))

mult_classes <- apply(mult_pred, 2, function(x) ifelse(x>median(x), 1, 0))

mult_tables <- vector(mode = "list", 5)
mult_tables[[1]] <- table(mult_classes[,"Adoption"], validation$Adoption)
mult_tables[[2]] <- table(mult_classes[,"Died"], validation$Died)
mult_tables[[3]] <- table(mult_classes[,"Euthanasia"], validation$Euthanasia)
mult_tables[[4]] <- table(mult_classes[,"Return_to_owner"], validation$Return_to_owner)
mult_tables[[5]] <- table(mult_classes[,"Transfer"], validation$Transfer)
mult_tables
mult_accuracy <- sapply(mult_tables, function(x) sum(diag(x))/sum(x))
mult_accuracy

xgb_pred
mult_pred
combined_preds <- (xgb_pred + mult_pred)/2
MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(xgb_pred))
MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(mult_pred))
MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(combined_preds))

png(filename = "scorings_histograms.png", width = 1024, height = 768)
par(mfrow=c(5,3))
hist(xgb_pred$Adoption)
hist(mult_pred$Adoption)
hist(final_pred$Adoption)
hist(xgb_pred$Died)
hist(mult_pred$Died)
hist(final_pred$Died)
hist(xgb_pred$Euthanasia)
hist(mult_pred$Euthanasia)
hist(final_pred$Euthanasia)
hist(xgb_pred$Return_to_owner)
hist(mult_pred$Return_to_owner)
hist(final_pred$Return_to_owner)
hist(xgb_pred$Transfer)
hist(mult_pred$Transfer)
hist(final_pred$Transfer)
dev.off()

plot(xgb_pred$Adoption, mult_pred$Adoption)
plot(xgb_pred$Died, mult_pred$Died)
plot(xgb_pred$Euthanasia, mult_pred$Euthanasia)
plot(xgb_pred$Return_to_owner, mult_pred$Return_to_owner)
plot(xgb_pred$Transfer, mult_pred$Transfer)
