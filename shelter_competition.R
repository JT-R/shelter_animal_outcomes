library(data.table)
library(xgboost)
library(nnet)
library(randomForest)

LoadData <- function(dataset_path){
  dataset <- fread(dataset_path)
}

MultiLogLoss <- function(act, pred)
{
  eps = 1e-15;
  nr <- nrow(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(nrow(act))      
  return(ll);
}

CreateHasNameFeature <- function(dataset){
  dataset[, Has_Name:= factor(ifelse(Name!="", 1, 0))]
}

ImputeMedian <- function(x){
  replace(x, is.na(x), median(x, na.rm=TRUE))
}

CreateDaysUponOutcomeFeature <- function(dataset){
  number_of_units <- as.integer(substr(x = dataset$AgeuponOutcome, start = 1, stop = 1))
  units <- as.factor(gsub(pattern = "^[0-9]{1,3} ", replacement = "", x = dataset$AgeuponOutcome))
  units_multiplier <- numeric(nrow(dataset)) + 1
  units_multiplier[which(units %in% c("day", "days"))] <- 1
  units_multiplier[which(units %in% c("week", "weeks"))] <- 7
  units_multiplier[which(units %in% c("month", "months"))] <- 30.5
  units_multiplier[which(units %in% c("year", "years"))] <- 365.25
  units_multiplier[which(units == "")] <- NA
  
  dataset[,DaysUponOutcome:= as.numeric(number_of_units*units_multiplier)]
  # set(dataset, i = is.na(DaysUponOutcome), j = DaysUponOutcome, value = median(dataset$DaysUponOutcome, na.rm=TRUE))
  dataset[which(is.na(DaysUponOutcome)), DaysUponOutcome := median(DaysUponOutcome, na.rm=TRUE)]
  dataset[,DaysUponOutcome:= ImputeMedian(DaysUponOutcome)]
}

CreateSexFeature <- function(dataset){
  dataset[grepl(pattern = "Male", SexuponOutcome), Sex:= "Male"]
  dataset[grepl(pattern = "Female", SexuponOutcome), Sex:= "Female"]
  dataset[SexuponOutcome %in% c("Unknown", ""), Sex:= "Unknown"]
  dataset[,Sex:=factor(Sex)]
}

GroupRareBreeds <- function(dataset){
  n <- nrow(dataset)
  breed_share <- dataset[, .N, Breed][order(-N)]
  breed_share[, cumulated_share:= cumsum(N)/n]
  breed_share[, Fixed_Breed:= factor(ifelse(cumulated_share>0.75, "Unknown", Breed))]
  
  dataset <- merge(dataset, breed_share[,list(Breed, Fixed_Breed)], by = "Breed")
}

FactorizeCharacterFeatures <- function(dataset){
  dataset[, OutcomeType:= factor(OutcomeType)]
  dataset[, OutcomeType:= relevel(OutcomeType, ref = "Euthanasia")]
  dataset[, AnimalType:= factor(AnimalType)]
  dataset[, Color:= factor(Color)]
}

FitUnknownLevelsToModel <- function(dataset, model){
  dataset[!"%in%"(Fixed_Breed, model$xlevels$Fixed_Breed), Fixed_Breed:= "Unknown"]
  dataset[!"%in%"(FirstColor, model$xlevels$FirstColor), FirstColor:= "Unknown"]
  dataset[!"%in%"(SecondColor, model$xlevels$SecondColor), SecondColor:= "Unknown"]
}

CreateTimeOfTheDayFeature <- function(dataset){
  dataset[, DateTime:= as.POSIXct(DateTime)]
  dataset[, Year:= as.factor(year(DateTime))]
  dataset[, Month:= as.factor(month(DateTime))]
  dataset[, Day:= as.factor(wday(DateTime))]
  dataset[, Hour:= as.integer(hour(DateTime))]
  dataset[, Season:= as.factor(quarter(DateTime))]
  
  dataset[Hour %in% c(0:6,20:23), TimeOfTheDay:= "Night"]
  dataset[Hour %in% c(7:19), TimeOfTheDay:= "Day"]
  dataset[, TimeOfTheDay:= as.factor(TimeOfTheDay)]
}

CreateIsDomesticFeature <- function(dataset){
  dataset[, IsDomestic:= as.factor(grepl(pattern = "Domestic", x = Breed))]
}

CreateIsMixFeature <- function(dataset){
  dataset[, IsMix:= as.factor(grepl(pattern = "Mix", x = Breed))]
}

CreateIsSterilizedFeature <- function(dataset){
  dataset[grepl(pattern = "Spayed|Neutered", x = SexuponOutcome), IsSterilized:= "TRUE"]
  dataset[grepl(pattern = "Intact", x = SexuponOutcome), IsSterilized:= "FALSE"]
  dataset[SexuponOutcome %in% c("", "Unknown"), IsSterilized:= "Unknown"]
  dataset[, IsSterilized:= as.factor(IsSterilized)]
}

CreateHairLengthFeature <- function(dataset){
  dataset[grepl(pattern = "Short", x = Breed), HairLength:= "Short"]
  dataset[grepl(pattern = "Medium", x = Breed), HairLength:= "Medium"]
  dataset[grepl(pattern = "Long", x = Breed), HairLength:= "Long"]
  dataset[is.na(HairLength), HairLength:= "Other"]
  dataset[, HairLength:= as.factor(HairLength)]
}

CreateNumberOfColorsFeature <- function(dataset){
  dataset[grepl(pattern = "/", x = Color, fixed = TRUE), NumberOfColors:= 2]
  dataset[Color == "Tricolor", NumberOfColors:= 3]
  dataset[is.na(NumberOfColors), NumberOfColors:= 1]
}

CreateFirstColorFeature <- function(dataset){
  dataset[, FirstColor:= factor(gsub(pattern = "/(.)*", replacement = "", x = Color))]
}

CreateSecondColorFeature <- function(dataset){
  dataset[, SecondColor:= factor(gsub(pattern = "(.)*/", replacement = "", x = Color))]
}

AddBinarizedOutcomeColumns <- function(dataset){
  binarized_outcome <- data.table(model.matrix(~ OutcomeType -1, dataset))
  setnames(binarized_outcome, colnames(binarized_outcome), gsub(pattern = "OutcomeType", replacement = "", x = colnames(binarized_outcome)))
  dataset <- data.table(cbind(dataset, binarized_outcome))
  
  return(dataset)
}

PreprocessDatasets <- function(labeled_data, testing){
  CreateHasNameFeature(labeled_data)
  labeled_data <- CreateDaysUponOutcomeFeature(labeled_data)
  CreateSexFeature(labeled_data)
  labeled_data <- GroupRareBreeds(labeled_data)
  FactorizeCharacterFeatures(labeled_data)
  CreateTimeOfTheDayFeature(labeled_data)
  CreateIsDomesticFeature(labeled_data)
  CreateIsMixFeature(labeled_data)
  CreateIsSterilizedFeature(labeled_data)
  CreateHairLengthFeature(labeled_data)
  CreateNumberOfColorsFeature(labeled_data)
  CreateFirstColorFeature(labeled_data)
  CreateSecondColorFeature(labeled_data)
  
  labeled_data <- AddBinarizedOutcomeColumns(labeled_data)
  
  CreateHasNameFeature(testing)
  testing <- CreateDaysUponOutcomeFeature(testing)
  CreateSexFeature(testing)
  testing <- GroupRareBreeds(testing)
  CreateTimeOfTheDayFeature(testing)
  CreateIsDomesticFeature(testing)
  CreateIsMixFeature(testing)
  CreateIsSterilizedFeature(testing)
  CreateHairLengthFeature(testing)
  CreateNumberOfColorsFeature(testing)
  CreateFirstColorFeature(testing)
  CreateSecondColorFeature(testing)
  
  return(list(labeled_data, testing))
}


labeled_data <- LoadData("train.csv")
testing <- LoadData("test.csv")
sample_submission <- LoadData("sample_submission.csv")




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
model <- multinom(OutcomeType ~ Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed + TimeOfTheDay +
                    Season + Year + Month + Day + IsMix + IsDomestic + Sex + IsSterilized + HairLength +
                    NumberOfColors + FirstColor, data = labeled_data)
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








n <- nrow(labeled_data)
training_indices <- sample(n, floor(n*2/3))
testing_indices <- setdiff(1:n, training_indices)

training <- labeled_data[training_indices]
validation <- labeled_data[testing_indices]

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



model <- multinom(OutcomeType ~ Sex + Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed, data = training)

model <- multinom(OutcomeType ~ Has_Name + DaysUponOutcome + AnimalType + Fixed_Breed + TimeOfTheDay +
                    Season + Year + Month + Day + IsMix + IsDomestic + Sex + IsSterilized + HairLength +
                    NumberOfColors + FirstColor, data = training)
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
