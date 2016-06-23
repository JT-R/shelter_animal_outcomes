# [1] "DateTime"           "OutcomeType"        "AnimalType"         "SexuponOutcome"    
# [5] "AgeuponOutcome"     "IsDomestic"         "IsSterilized"       "Hour"              
# [9] "Weekday"            "NameLen"            "NameWeirdness"      "breed.Pit.Bull.Mix"


PreprocessDatasets <- function(labeled_data, testing){
  # CreateHasNameFeature(labeled_data)
  # labeled_data <- CreateDaysUponOutcomeFeature(labeled_data)
   CreateSexFeature(labeled_data)
  # CreateFirstBreedFeature(labeled_data)
  # labeled_data <- GroupRareBreeds(labeled_data)
  # FactorizeCharacterFeatures(labeled_data)
  #CreateTimeOfTheDayFeature(labeled_data)
  CreateIsDomesticFeature(labeled_data)
  CreateIsMixFeature(labeled_data)
  CreateIsSterilizedFeature(labeled_data)
  # CreateHairLengthFeature(labeled_data)
  # CreateNumberOfColorsFeature(labeled_data)
  # CreateFirstColorFeature(labeled_data)
  # CreateSecondColorFeature(labeled_data)
  # CreateAgeCategoriesFeature(labeled_data)
  # CreateIsWeekendFeature(labeled_data)
  # CreateYearPartFeature(labeled_data)
  # subtypes_per_breed <- CreateBreedOutputSubtypesProbabilityFeatures(labeled_data)
  # labeled_data <- merge(labeled_data, subtypes_per_breed, by = "Fixed_Breed")
  
  # labeled_data <- AddBinarizedOutcomeColumns(labeled_data)
  
  # CreateHasNameFeature(testing)
  # testing <- CreateDaysUponOutcomeFeature(testing)
   CreateSexFeature(testing)
  # CreateFirstBreedFeature(testing)
  # testing <- GroupRareBreeds(testing)
  # FactorizeCharacterFeatures(testing)
  #CreateTimeOfTheDayFeature(testing)
  CreateIsDomesticFeature(testing)
  CreateIsMixFeature(testing)
  CreateIsSterilizedFeature(testing)
  # CreateHairLengthFeature(testing)
  # CreateNumberOfColorsFeature(testing)
  # CreateFirstColorFeature(testing)
  # CreateSecondColorFeature(testing)
  # CreateAgeCategoriesFeature(testing)
  # CreateIsWeekendFeature(testing)
  # CreateYearPartFeature(testing)
  # testing <- merge(testing, subtypes_per_breed, by = "Fixed_Breed", all.x = TRUE)
  
  return(list(labeled_data = labeled_data,
              testing = testing))
}

labeled_set <- data.table(read.csv("train.csv"))
testing <- data.table(read.csv("test.csv"))
n <- nrow(labeled_set)
popularBreeds <- names(summary(labeled_set$Breed,maxsum=50L))
trainNameSummary <- summary(labeled_set$Name,maxsum=Inf)

processed_datasets <- PreprocessDatasets(labeled_set, testing)
labeled_set <- processed_datasets[['labeled_data']]
#labeled_set[, SexuponOutcome:= NULL]
testing <- processed_datasets[['testing']]

validation_datasets <- CreateValidationDatasets(labeled_set, n)
training <- validation_datasets[['training']]
validation <- validation_datasets[['validation']]

training <- clean(data.frame(labeled_set))
#training <- clean(training)
validation <- clean(data.frame(validation))
testing <- clean(data.frame(testing))
actual_classes <- data.table(model.matrix(~ OutcomeType -1, data = validation))

#training[, Hour:= NULL]

for(i in names(training)) if(is.logical(training[[i]])) {training[[i]] <- as.numeric(training[[i]]); validation[[i]] <- as.numeric(validation[[i]])}

predictors <- colnames(training)
predictors <- setdiff(c(predictors, "OutcomeType"),
                      c("DateTime","Euthanasia", "Adoption", "Died", "Return_to_owner", "Transfer"))

set.seed(20160324L)
gbm_model <- gbm(
  OutcomeType ~ .,
  # data=training[,predictors, with=FALSE],
  data = training,
  distribution="multinomial",
  shrinkage=0.05,
  n.trees=500,
  interaction.depth=6L,
#  train.fraction=0.8,
  keep.data=FALSE,
  verbose=TRUE
)

summary(gbm_model)
nonimportant_features <- rownames(summary(gbm_model))[which(summary(gbm_model)[,2] < 0.070)]
training <- training[ , !(names(training) %in% nonimportant_features)]
training <- training[ , (names(training) %in% only_features)]


testPreds2 <- predict(gbm_model,validation,type="response")
dim(testPreds2) <- c(nrow(validation),5)
colnames(testPreds2) <- levels(training$OutcomeType)

MultiLogLoss(act = as.matrix(actual_classes), pred = as.matrix(testPreds2))

submission_preds <- predict(gbm_model, testing, type = "response", n.trees = 500)
dim(submission_preds) <- c(nrow(submission_preds),5)
colnames(submission_preds) <- levels(training$OutcomeType)
submission_preds <- cbind(ID = testing$ID, submission_preds)

write.csv(x = submission_preds, file = "simpler_predictions_v2.csv", row.names=FALSE, quote=FALSE)
