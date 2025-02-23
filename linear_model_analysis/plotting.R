# First, let's load the required libraries
library(ggplot2)
library(dplyr)

df <- read.csv("data/combined.csv")








# Country × Modifier Interaction
ggplot(df, aes(x = modifier, y = response_difference_n, fill = country)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(0.9), width = 0.25) +
  labs(title = "Country × Modifier Interaction",
       x = "Modifier",
       y = "Response Difference",
       fill = "Country") +
  theme_minimal()

ggsave("figures/country_modifier_interaction.pdf", width = 6, height = 6)



# Predicate × Politeness Interaction
ggplot(df, aes(x = politeness_difference, y = response_difference_n, color = modifier)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ predicate, scales = "free_y") +
  labs(title = "Predicate × Politeness Interaction",
       x = "Politeness Difference",
       y = "Response Difference",
       color = "Modifier") +
  theme_minimal()






# Calculate means for each scenario and country
scenario_means <- df %>%
  group_by(country, predicate, modifier) %>%
  summarise(
    mean_politeness = mean(politeness_difference),
    mean_response = mean(response_difference_n)
  ) %>%
  mutate(scenario = paste(modifier, predicate)) %>%
  tidyr::pivot_longer(
    cols = c(mean_politeness, mean_response),
    names_to = "measure",
    values_to = "value"
  )
# First, calculate the positions for the separator lines
separator_positions <- scenario_means %>%
  select(modifier, scenario) %>%
  distinct() %>%
  group_by(modifier) %>%
  summarise(n = n()/2) %>%  # Divide by 2 since each scenario has 2 measures
  mutate(position = cumsum(n) + 0.5) %>%
  # Remove the last line (after "very")
  slice(1:(n()-1))

# Modify the plot to add separator lines
ggplot(scenario_means, aes(x = value, y = scenario, fill = measure)) +
  # Add separator lines
  geom_hline(yintercept = separator_positions$position, 
             color = "gray70",
             linetype = "dashed",
             linewidth = 0.5,
             linetype = "22") +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~country, ncol = 2) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(hjust = 1),
    legend.title = element_blank(),
    axis.text.y.right = element_blank(),
    axis.title.y.right = element_blank(),
    panel.spacing = unit(1, "lines"),
    plot.title = element_text(hjust = 0.5),
  ) +
  labs(
    y = "Scenario (Modifier + Predicate)",
    x = "Average Difference",
    title = "Average Politeness and Meaning Differences by Scenario and Country"
  ) +
  scale_fill_manual(
    values = c("mean_politeness" = "skyblue", "mean_response" = "coral"),
    labels = c("mean_politeness" = "Politeness Difference", 
               "mean_response" = "Meaning Difference")
  )

ggsave("comparison_by_modifier.pdf", width = 12, height = 8)






# Calculate means for each scenario and country
scenario_means <- df %>%
  group_by(country, predicate, modifier) %>%
  summarise(
    mean_politeness = mean(politeness_difference),
    mean_response = mean(response_difference_n)
  ) %>%
  mutate(scenario = paste(predicate, modifier)) %>%  # Changed order here
  tidyr::pivot_longer(
    cols = c(mean_politeness, mean_response),
    names_to = "measure",
    values_to = "value"
  )

# Calculate separator positions
separator_positions <- scenario_means %>%
  select(predicate, scenario) %>%  # Changed from modifier to predicate
  distinct() %>%
  group_by(predicate) %>%         # Group by predicate instead of modifier
  summarise(n = n()/2) %>%
  mutate(position = cumsum(n) + 0.5) %>%
  slice(1:(n()-1))


# Modify the plot to add separator lines
ggplot(scenario_means, aes(x = value, y = scenario, fill = measure)) +
  # Add separator lines
  geom_hline(yintercept = separator_positions$position, 
             color = "gray70",
             linetype = "dashed",
             linewidth = 0.5,
             linetype = "22") +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~country, ncol = 2) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(hjust = 1),
    legend.title = element_blank(),
    axis.text.y.right = element_blank(),
    axis.title.y.right = element_blank(),
    panel.spacing = unit(1, "lines"),
    plot.title = element_text(hjust = 0.5),
  ) +
  labs(
    y = "Scenario (Modifier + Predicate)",
    x = "Average Difference",
    title = "Average Politeness and Meaning Differences by Scenario and Country"
  ) +
  scale_fill_manual(
    values = c("mean_politeness" = "skyblue", "mean_response" = "coral"),
    labels = c("mean_politeness" = "Politeness Difference", 
               "mean_response" = "Meaning Difference")
  )

ggsave("comparison_by_predicate.pdf", width = 12, height = 8)
