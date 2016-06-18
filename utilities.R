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

FactorizeCharacterFeatures <- function(dataset){
  dataset[, Breed:= factor(Breed)]
  dataset[, Name:= factor(Name)]
  if("OutcomeType" %in% colnames(dataset)){
    dataset[, OutcomeType:= factor(OutcomeType)]
    dataset[, OutcomeType:= relevel(OutcomeType, ref = "Euthanasia")]
  }
  dataset[, AnimalType:= factor(AnimalType)]
  dataset[, Color:= factor(Color)]
  dataset[, SexuponOutcome:= factor(SexuponOutcome)]
  dataset[, AgeuponOutcome:= factor(AgeuponOutcome)]
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
  
  dataset[Hour %in% c(0:8,17:23), TimeOfTheDay:= "Night"]
  dataset[Hour %in% c(9), TimeOfTheDay:= "Opening"]
  dataset[Hour %in% c(10:16), TimeOfTheDay:= "Day"]
  dataset[, TimeOfTheDay:= as.factor(TimeOfTheDay)]
}

CreateIsDomesticFeature <- function(dataset){
  dataset[, IsDomestic:= as.factor(grepl(pattern = "Domestic", x = Breed))]
}

CreateIsMixFeature <- function(dataset){
  dataset[, IsMix:= as.factor(grepl(pattern = "Mix|/", x = Breed))]
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

CreateAgeCategoriesFeature <- function(dataset){
  dataset[DaysUponOutcome<=28, AgeCategory:="Young"]
  dataset[DaysUponOutcome>28 & DaysUponOutcome<274.50, AgeCategory:="Medium"]
  dataset[DaysUponOutcome>=274.50, AgeCategory:="Old"]
  
  dataset[, AgeCategory:= as.factor(AgeCategory)]
}

CreateIsWeekendFeature <- function(dataset){
  dataset[Day %in% c("1","7"), IsWeekend:= TRUE]
  dataset[Day %in% c("2","3","4","5","6"), IsWeekend:= FALSE]
  dataset[,IsWeekend:= factor(IsWeekend)]
}

CreateYearPartFeature <- function(dataset){
  dataset[Month %in% c("1", "2", "11", "12"), YearPart:= "1_2_11_12"]
  dataset[Month %in% c("3", "4", "5", "6"), YearPart:= "3_4_5_6"]
  dataset[Month %in% c("7", "8", "9", "10"), YearPart:= "7_8_9_10"]
  dataset[, YearPart:= factor(YearPart)]
}

CreateFirstBreedFeature <- function(dataset){
  dataset[, FirstBreed:= factor(gsub(pattern = "/(.)*", replacement = "", Breed))]
}

GroupRareBreeds <- function(dataset){
  n <- nrow(dataset)
  breed_share <- dataset[, .N, Breed][order(-N)]
  breed_share[, cumulated_share:= cumsum(N)/n]
  breed_share[cumulated_share<0.75, Fixed_Breed:= Breed]
  breed_share[cumulated_share>0.75, Fixed_Breed:= "Unknown"]
  # breed_share[, Fixed_Breed:= factor(ifelse(cumulated_share>0.75, "Unknown", FirstBreed))]
  
  dataset <- merge(dataset, breed_share[,list(Breed, Fixed_Breed)], by = "Breed")
  dataset[, Fixed_Breed:= factor(Fixed_Breed)]
}

PreprocessDatasets <- function(labeled_data, testing){
  CreateHasNameFeature(labeled_data)
  labeled_data <- CreateDaysUponOutcomeFeature(labeled_data)
  CreateSexFeature(labeled_data)
  CreateFirstBreedFeature(labeled_data)
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
  CreateAgeCategoriesFeature(labeled_data)
  CreateIsWeekendFeature(labeled_data)
  CreateYearPartFeature(labeled_data)
  
  labeled_data <- AddBinarizedOutcomeColumns(labeled_data)
  
  CreateHasNameFeature(testing)
  testing <- CreateDaysUponOutcomeFeature(testing)
  CreateSexFeature(testing)
  CreateFirstBreedFeature(testing)
  testing <- GroupRareBreeds(testing)
  FactorizeCharacterFeatures(testing)
  CreateTimeOfTheDayFeature(testing)
  CreateIsDomesticFeature(testing)
  CreateIsMixFeature(testing)
  CreateIsSterilizedFeature(testing)
  CreateHairLengthFeature(testing)
  CreateNumberOfColorsFeature(testing)
  CreateFirstColorFeature(testing)
  CreateSecondColorFeature(testing)
  CreateAgeCategoriesFeature(testing)
  CreateIsWeekendFeature(testing)
  CreateYearPartFeature(testing)
  
  return(list(labeled_data = labeled_data,
              testing = testing))
}

CreateValidationDatasets <- function(training, validation, seed = 123){
  n <- nrow(labeled_data)
  training_indices <- sample(n, floor(n*2/3))
  testing_indices <- setdiff(1:n, training_indices)
  
  training <- labeled_data[training_indices]
  validation <- labeled_data[testing_indices]
  
  return(list(training = training,
              validation = validation))
}

