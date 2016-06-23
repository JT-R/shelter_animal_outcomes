library(data.table)

labeled_data

labeled_data[,.N, Fixed_Breed]
labeled_data[,.N, Color]


grouped <- labeled_data[,list(Adoption = mean(Adoption),
                             Euthanasia = mean(Euthanasia),
                             Died = mean(Died),
                             Return_to_owner = mean(Return_to_owner),
                             Transfer = mean(Transfer),
                             N = .N), by = list(OutcomeSubtype)][order(OutcomeSubtype)][N>400]

grouped
feature <- grouped$Hour
with(grouped, plot(feature, Adoption, type = "p", pch = 18, ylim = c(0,1)))
with(grouped, points(feature, Euthanasia, col = "red"))
with(grouped, points(feature, Died, col = "green"))
with(grouped, points(feature, Return_to_owner, col = "blue"))
with(grouped, points(feature, Transfer, col = "yellow"))

labeled_data[,list(Adoption = mean(Adoption),
                   Euthanasia = mean(Euthanasia),
                   Died = mean(Died),
                   Return_to_owner = mean(Return_to_owner),
                   Transfer = mean(Transfer),
                   N = .N), by = list(Fixed_Breed)][order(Adoption)]
