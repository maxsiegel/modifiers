# Load required libraries
library(tidyverse)
library(DT)

# Experiment 1 data
# An experiment where participants were asked to interpret various statements with and without an intensifier
UK_df <- read.csv("data/UK_df.csv")
US_df <- read.csv("data/US_df.csv")
# combine data
combined_df <- rbind(UK_df, US_df)
# filter out rows where there is no intensifier (within subject design)
combined_df <- combined_df %>%
  filter(has.intensifier. != "no")
# rename columns
combined_df <- combined_df %>%
  rename(interpretation_difference = Z.Score.Difference)
# add column for experiment
combined_df <- combined_df %>%
  mutate(experiment = "Interpretation")

# Experiment 2 data
# An experiment where participants were asked to interpret various statements with and without an intensifier (narrator version)
UK_narrator_df <- read.csv("data/UK_narrator_df.csv")
US_narrator_df <- read.csv("data/US_narrator_df.csv")
# combine data
combined_narrator_df <- rbind(UK_narrator_df, US_narrator_df)
# filter out rows where there is no intensifier (within subject design)
combined_narrator_df <- combined_narrator_df %>%
  filter(has.intensifier. != "no")
# rename columns
# add column for experiment
combined_narrator_df <- combined_narrator_df %>%
  mutate(experiment = "Politeness_Narrator")

combined_narrator_df <- combined_narrator_df %>%
  rename(interpretation_difference = Z.Score.Difference)

# adding an item id column
combined_narrator_df_clean <- combined_narrator_df %>%
  mutate(scenario = paste(intensifier, predicate, sep = "."))
# rename columns
combined_narrator_df_clean <- combined_narrator_df_clean %>%
  rename(modifier = intensifier)
write.csv(combined_narrator_df_clean, "data/df_narrator_clean.csv", row.names = FALSE)

# average politeness ratings for narrator condition
combined_narrator_df <- combined_narrator_df %>%
  group_by(country, predicate, intensifier) %>%
  summarise(narrator_interpretation = mean(interpretation_difference))

# Experiment 3 data
# An experiment where participants were asked to rate the politeness of various statements with and without an intensifier (direct version)
UK_direct_df <- read.csv("data/UK_direct_df.csv")
US_direct_df <- read.csv("data/US_direct_df.csv")
# combine data
combined_direct_df <- rbind(UK_direct_df, US_direct_df)
# filter out rows where there is no intensifier (within subject design)
combined_direct_df <- combined_direct_df %>%
  filter(has.intensifier. != "no")
# add column for experiment
combined_direct_df <- combined_direct_df %>%
  mutate(experiment = "Politeness_Direct")
# average politness ratings
combined_direct_df <- combined_direct_df %>%
  group_by(country, predicate, intensifier) %>%
  summarise(politeness_difference = mean(Z.Score.Difference))


# combine interpretation and politeness direct data
df <- inner_join(combined_df, combined_narrator_df, 
                 by = c("country", "intensifier", "predicate"))
df <- inner_join(df, combined_direct_df, 
                 by = c("country", "intensifier", "predicate"))
# adding an item id column
df <- df %>%
  mutate(scenario = paste(intensifier, predicate, sep = "."))
# rename columns
df <- df %>%
  rename(modifier = intensifier)

colnames(df)
write.csv(df, "data/combined.csv", row.names = FALSE)


