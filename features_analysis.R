library(data.table)

labeled_data

labeled_data[,.N, Fixed_Breed]
labeled_data[,.N, Color]

labeled_data[,list(Adoption = mean(Adoption),
                   Euthanasia = mean(Euthanasia),
                   Died = mean(Died),
                   Return_to_owner = mean(Return_to_owner),
                   Transfer = mean(Transfer),
                   N = .N), by = list(HairLength)]

labeled_data[,list(Adoption = mean(Adoption),
                   Euthanasia = mean(Euthanasia),
                   Died = mean(Died),
                   Return_to_owner = mean(Return_to_owner),
                   Transfer = mean(Transfer),
                   N = .N), by = list(Fixed_Breed)][order(Adoption)]
