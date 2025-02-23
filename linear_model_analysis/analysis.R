# Setup 
library(lme4)
library(lmerTest)
library(tidyverse)
library(DT)
library(simr)
library(ggeffects)
library(cowplot)
library(broom.mixed)
library(MuMIn)


# Modifiers used in the study: extremely, kind of, quite, slightly, very (strongest effects for "quite" and "very", "slightly" weaker
# Candidates for new study: quite, very, absolutely, a bit, really, totally, so (?), awfully (irony and humor)

# Data Loading 
df <- read.csv("data/combined.csv")
# Experiment 1
# Figure 1: Interpretation Differences by Country and Modifier

# 1. Descriptive Statistics
summary_stats_sorted <- summary_stats %>%
  # Get UK means to sort by
  group_by(modifier) %>%
  mutate(uk_mean = mean_diff[country == "UK"]) %>%
  ungroup() %>%
  # Convert modifier to factor with levels ordered by UK means
  mutate(modifier = factor(modifier, 
                          levels = modifier[country == "UK"][order(uk_mean[country == "UK"])]))

# 2. Create and save the plot
interpretation_plot <- ggplot(summary_stats_sorted, 
       aes(x = modifier, y = mean_diff, fill = country)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = mean_diff - se, ymax = mean_diff + se),
                position = position_dodge(0.9), width = 0.2) +
  theme_minimal() +
  labs(title = "Interpretation Differences by Country and Modifier",
       y = "Mean Interpretation Difference",
       x = "Modifier") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Angle labels for better readability

# show the plot
interpretation_plot

# Save the plot
ggsave("figures/interpretation_differences.pdf", 
       interpretation_plot, 
       width = 6, 
       height = 5, 
       units = "in")

# 3. Mixed Effects Model Analysis
base_model <- lmer(interpretation_difference ~ 
           country + modifier + predicate +
           (1|person_id) + (1|scenario), 
           data = df)
summary(base_model)


interaction_model <- lmer(interpretation_difference ~ 
           country * modifier + predicate +
           (1|person_id) + (1|scenario), 
           data = df)
summary(interaction_model)
anova(base_model, interaction_model)

# 4. Post-hoc analysis for specific modifiers
library(emmeans)
emmeans_results <- emmeans(interaction_model, pairwise ~ country | modifier)
emmeans_results




# Valence analysis
df <- df %>%
  mutate(valence = case_when(
    predicate %in% c("understandable", "impressive", "helpful") ~ "positive",
    predicate %in% c("exhausted", "boring", "difficult", "concerned") ~ "negative",
    TRUE ~ NA_character_
  ))

modifier_order <- df %>%
  group_by(modifier) %>%
  summarise(avg_strength = mean(interpretation_difference)) %>%
  arrange(avg_strength) %>%
  pull(modifier)


# Create summary statistics grouped by modifier
summary_stats_valence <- df %>%
  group_by(modifier, country, valence) %>%
  summarise(
    mean_diff = mean(interpretation_difference),
    sd_diff = sd(interpretation_difference),
    n = n(),
    se = sd_diff/sqrt(n),
    .groups = 'drop'
  ) %>%
  # Create combined factor for valence and country
  mutate(
    country_valence = paste(valence, country),  # Changed order here
    # Order the country_valence factor
    country_valence = factor(country_valence, 
                           levels = c("negative UK", "negative US", 
                                    "positive UK", "positive US")),
    # Set modifier order based on average strength
    modifier = factor(modifier, levels = modifier_order)
  )

# Create the plot
valence_plot <- ggplot(summary_stats_valence, 
       aes(x = country_valence, y = mean_diff, fill = country)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = mean_diff - se, ymax = mean_diff + se),
                position = position_dodge(0.9), width = 0.2) +
  facet_wrap(~modifier, ncol = 5) +  # or however many columns you prefer
  theme_minimal() +
  labs(title = "Interpretation Differences by Modifier, Country, and Predicate Valence",
       y = "Mean Interpretation Difference",
       x = "Country and Valence") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

valence_plot
# Save the plot
ggsave("figures/interpretation_differences_by_modifier.pdf", 
       valence_plot, 
       width = 12, 
       height = 6, 
       units = "in")



# 4. Update the mixed effects model to include valence
valence_model <- lmer(interpretation_difference ~ 
                     country * modifier * valence + 
                     (1|person_id) + (1|scenario), 
                     data = df)


# 5. Compare models
anova(base_model, valence_model)

# 6. Get detailed model summary
summary(valence_model)

# 7. Post-hoc analyses
library(emmeans)
# Look at country differences by modifier and valence
emmeans_valence <- emmeans(valence_model, pairwise ~ country | modifier | valence)

# Look at valence effects within each modifier
emmeans_modifier <- emmeans(valence_model, pairwise ~ valence | modifier)

# Print results
print(emmeans_valence)
print(emmeans_modifier)






# Modify titles to be (a) and (b) instead of full titles
interpretation_plot <- ggplot(summary_stats_sorted, 
       aes(x = modifier, y = mean_diff, fill = country)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = mean_diff - se, ymax = mean_diff + se),
                position = position_dodge(0.9), width = 0.2) +
  theme_minimal() +
  labs(title = "(a) Overall Interpretation Differences",
       y = "Mean Interpretation Difference",
       x = "Modifier") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Create the plot
valence_plot <- ggplot(summary_stats_valence, 
       aes(x = country_valence, y = mean_diff, fill = country)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = mean_diff - se, ymax = mean_diff + se),
                position = position_dodge(0.9), width = 0.2) +
  facet_wrap(~modifier, ncol = 5) +  # or however many columns you prefer
  theme_minimal() +
  labs(title = "Interpretation Differences by Modifier, Country, and Predicate Valence",
       y = "Mean Interpretation Difference",
       x = "Country and Valence") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Combine plots
combined_plot <- plot_grid(
  interpretation_plot, valence_plot,
  ncol = 2,  # Two columns
  rel_widths = c(1, 1.5)  # Make valence plot slightly wider
)

combined_plot

# Save the combined plot
ggsave("figures/combined_interpretation_differences.pdf", 
       combined_plot, 
       width = 15,  # Increased width for better visibility
       height = 6, 
       units = "in")







# Experiment 2: Politeness
df_narrator <- read.csv("data/df_narrator_clean.csv")

# 1. Descriptive Statistic
# First calculate summary statistics with mean_diff
summary_stats_narrator <- df_narrator %>%
  group_by(country, modifier) %>%
  summarise(
    mean_diff = mean(interpretation_difference),  # or whatever your raw score column is called
    sd_diff = sd(interpretation_difference),
    n = n(),
    se = sd_diff/sqrt(n),
    .groups = 'drop'
  )

# Then sort by UK means
summary_stats_narrator_sorted <- summary_stats_narrator %>%
  # Get UK means to sort by
  group_by(modifier) %>%
  mutate(uk_mean = mean_diff[country == "UK"]) %>%
  ungroup() %>%
  # Convert modifier to factor with levels ordered by UK means
  mutate(modifier = factor(modifier, 
                          levels = modifier[country == "UK"][order(uk_mean[country == "UK"])]))

# 2. Create and save the plot
interpretation_plot <- ggplot(summary_stats_narrator_sorted, 
       aes(x = modifier, y = mean_diff, fill = country)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = mean_diff - se, ymax = mean_diff + se),
                position = position_dodge(0.9), width = 0.2) +
  theme_minimal() +
  labs(title = "Interpretation Differences by Country and Modifier",
       y = "Mean Interpretation Difference",
       x = "Modifier") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Angle labels for better readability


# show the plot
interpretation_plot

# Save the plot
# ggsave("figures/interpretation_differences.pdf", 
#        interpretation_plot, 
#        width = 6, 
#        height = 5, 
#        units = "in")



# politeness effects are not uniform across predicates but vary by the specific predicate used

summary(df)

politeness_model <- lmer(interpretation_difference ~ 
           country * modifier + predicate + politeness_difference +
           (1|person_id) + (1|scenario), 
           data = df)
summary(politeness_model)


interaction_model <- lmer(interpretation_difference ~ 
           country * modifier + predicate +
           (1|person_id) + (1|scenario), 
           data = df)
summary(interaction_model)

anova(interaction_model, politeness_model)

# Look at coefficients from both models
int_coef = summary(interaction_model)$coefficients
pol_coef = summary(politeness_model)$coefficients

# Create comparison table
mediation_effects <- data.frame(
  modifier = c("kind of", "quite", "slightly", "very"),
  original_effect = int_coef[grep("countryUS:modifier", rownames(int_coef)), "Estimate"],
  original_p = int_coef[grep("countryUS:modifier", rownames(int_coef)), "Pr(>|t|)"],
  controlled_effect = pol_coef[grep("countryUS:modifier", rownames(pol_coef)), "Estimate"],
  controlled_p = pol_coef[grep("countryUS:modifier", rownames(pol_coef)), "Pr(>|t|)"]
) %>%
  mutate(
    reduction = abs(original_effect) - abs(controlled_effect),
    percent_reduction = (reduction / abs(original_effect)) * 100
  )

print(mediation_effects)


# ggplot(mediation_effects, aes(x = reorder(modifier, reduction))) +
#   geom_segment(aes(y = 0, yend = reduction, xend = modifier), 
#               color = "gray50") +
#   geom_point(aes(y = reduction), color = "coral", size = 3) +
#   geom_text(aes(y = reduction, 
#                 label = paste0(round(percent_reduction, 1), "%\n(", round(original_effect, 3), 
#                              " → ", round(controlled_effect, 3), ")")), 
#             vjust = -1) +
#   theme_minimal() +
#   labs(
#     title = "Reduction in US-UK Differences After Controlling for Narrator Interpretation",
#     subtitle = "Values show coefficients before → after controlling for narrator interpretation",
#     y = "Reduction in |US-UK Difference|",
#     x = "Modifier"
#   ) +
#   coord_flip()



# does not converge
# politeness_model_complex <- lmer(interpretation_difference ~ 
#            country * modifier + predicate * narrator_interpretation +
#            (1|person_id) + (1|scenario), 
#            data = df)
# summary(politeness_model_complex)
# anova(interaction_model, politeness_model_complex)





# Calculate variance components for each model
interaction_var <- as.data.frame(VarCorr(interaction_model))
politeness_var <- as.data.frame(VarCorr(politeness_model))

# Calculate R-squared for both models
interaction_r2 <- r.squaredGLMM(interaction_model)
politeness_r2 <- r.squaredGLMM(politeness_model)

# Decompose variance using Type III ANOVA for each model
interaction_decomp <- anova(interaction_model, type = "III")
politeness_decomp <- anova(politeness_model, type = "III")

# Create a comparison table of variance explained by each term
library(broom.mixed)
interaction_components <- tidy(interaction_decomp) %>%
  mutate(model = "interaction")
politeness_components <- tidy(politeness_decomp) %>%
  mutate(model = "politeness")

# Combine and compare
variance_comparison <- bind_rows(interaction_components, politeness_components) %>%
  select(model, term, statistic, p.value, sumsq) %>%
  spread(model, sumsq) %>%
  mutate(unique_to_politeness = politeness - interaction)

# Visualize the variance components
ggplot(variance_comparison, aes(x = reorder(term, unique_to_politeness))) +
  geom_bar(aes(y = interaction), stat = "identity", fill = "blue", alpha = 0.5) +
  geom_bar(aes(y = unique_to_politeness), stat = "identity", fill = "red", alpha = 0.5) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variance Components by Model",
       subtitle = "Blue = Shared variance, Red = Unique to politeness model",
       x = "Model Terms",
       y = "Sum of Squares")



# Recalculate variance components with better matching
variance_comparison <- bind_rows(
  interaction_components %>% mutate(model = "interaction"),
  politeness_components %>% mutate(model = "politeness")
) %>%
  # Standardize term names to match between models
  mutate(term = gsub("country:modifier", "country_modifier", term)) %>%
  group_by(term) %>%
  summarize(
    interaction = sum(sumsq[model == "interaction"], na.rm = TRUE),
    politeness = sum(sumsq[model == "politeness"], na.rm = TRUE),
    unique_to_politeness = politeness - interaction
  ) %>%
  # Filter out terms with zero or negative differences
  filter(!is.na(unique_to_politeness))

# Create improved visualization
ggplot(variance_comparison, aes(x = reorder(term, unique_to_politeness))) +
  geom_bar(aes(y = interaction), stat = "identity", fill = "blue", alpha = 0.5) +
  geom_bar(aes(y = unique_to_politeness), stat = "identity", fill = "red", alpha = 0.5, position = "stack") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variance Components by Model",
       subtitle = "Blue = Shared variance, Red = Unique to politeness model",
       x = "Model Terms",
       y = "Sum of Squares")




# Calculate variance decomposition
interaction_decomp <- anova(interaction_model, type = "III")
politeness_decomp <- anova(politeness_model, type = "III")

variance_comparison <- bind_rows(
  tidy(interaction_decomp) %>% mutate(model = "interaction"),
  tidy(politeness_decomp) %>% mutate(model = "politeness")
) %>%
  mutate(term = gsub("country:modifier", "country_modifier", term)) %>%
  group_by(term) %>%
  summarize(
    interaction_total = sum(sumsq[model == "interaction"], na.rm = TRUE),
    politeness_total = sum(sumsq[model == "politeness"], na.rm = TRUE),
    shared = pmin(interaction_total, politeness_total),
    unique_interaction = interaction_total - shared,
    unique_politeness = politeness_total - shared
  ) %>%
  filter(!is.na(shared))

# Create improved visualization
ggplot(variance_comparison, aes(x = reorder(term, shared))) +
  geom_bar(aes(y = unique_interaction), stat = "identity", fill = "green", alpha = 0.5) +
  geom_bar(aes(y = shared), stat = "identity", fill = "blue", alpha = 0.5, position = "stack") +
  geom_bar(aes(y = unique_politeness), stat = "identity", fill = "red", alpha = 0.5, position = "stack") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variance Components by Model",
       subtitle = "Green = Unique to interaction model\nBlue = Shared variance\nRed = Unique to politeness model",
       x = "Model Terms",
       y = "Sum of Squares")


# Print the actual values
print(variance_comparison, n = Inf)

# Look at the model comparison statistics again
anova(interaction_model, politeness_model)







politeness_pred <- ggpredict(politeness_model, terms = c("modifier", "country", "politeness_difference"))
plot(politeness_pred) +
  labs(title = "Model Predictions: Politeness Effect by Modifier and Country",
       x = "Modifier",
       y = "Predicted Response Difference") +
  theme_minimal()


# TODO: Refining Predictors:
# Grouping predicates into broader categories (e.g., positive vs. negative sentiment, or task-oriented vs. emotional descriptors) may reveal stronger trends.
# Politeness ratings could also be standardized or transformed to reduce noise.


# Visualize Predicate-Specific Politeness Effects
library(ggeffects)
politeness_pred <- ggpredict(politeness_model, terms = c("politeness_difference", "predicate"))
plot(politeness_pred)




# fit separate models to establish the mediation effect
mediator_model <- lmer(politeness_difference ~ 
           country * modifier + predicate + 
           (1|scenario), 
           data = df)
summary(mediator_model)

outcome_model <- lmer(response_difference_n ~ 
           politeness_difference + predicate + 
           (1|person_id) + (1|scenario), 
           data = df)
summary(outcome_model)









# Base models
# m1: Base model with main effects only
# To test whether country, modifier, and predicate independently predict responses
m1 <- lmer(response_difference_n ~ 
           country + modifier + predicate +
           (1|person_id), 
           data = df)

# m2: Model with country x modifier interaction
# To test whether modifier interpretation varies by country
m2 <- lmer(response_difference_n ~ 
           country * modifier + predicate + 
           (1|person_id), 
           data = df)

# Politeness mediation models
# m3: Model with politeness as additional predictor
# To test whether perceived politeness differences explain (some of) the cultural variation in modifier interpretation
# TODO: explain why it's important that politeness is added as a *predicate-dependent* interaction  
m3 <- lmer(response_difference_n ~ 
           country * modifier + 
           politeness_difference * predicate +
           (1|person_id), 
           data = df)

# m3_full: Full model with predicate-specific politeness effects
# To test whether politeness effects vary by predicate valence
# while maintaining the mediation of cultural differences
m3_full <- lmer(response_difference_n ~ 
           country * modifier + 
           politeness_difference * modifier +  
           politeness_difference * predicate +
           (1|person_id), 
           data = df)

m3_threeway <- lmer(response_difference_n ~ 
           country * modifier * politeness_difference + 
           predicate +
           (1|person_id), 
           data = df)

# Analysis
summary(m1)
summary(m2) 
summary(m3)
summary(m3_full)
summary(m3_threeway)

# Model comparisons
# -----------------
# Test for country x modifier interaction
# Adding country×modifier interaction improves fit significantly (χ² = 15.628, p = 0.004)
# Cultural differences appear for "quite" (β = 0.296, p = 0.054) and "very" (β = -0.262, p = 0.088)
anova(m1, m2)  

# Test effect of adding politeness
# Adding politeness×predicate interactions improves fit substantially (χ² = 30.637, p < 0.0001)
# Differences become weaker when controlling for politeness×predicate
anova(m2, m3)  

# Compare full model sequence
# m3 vs m3_full: Adding politeness×modifier interactions doesn't improve fit significantly (χ² = 7.396, p = 0.116)
# In m3_full: We see significant politeness×modifier interactions for:
# "kind of" (β = 1.179, p = 0.010)
# "slightly" (β = 0.979, p = 0.030)
# "very" (β = 0.835, p = 0.085)
anova(m2, m3, m3_full)  

# These results suggest that:
# - There are cultural differences in modifier interpretation
# - These differences are partially explained by how politeness affects meaning differently across predicates
# - The relationship between politeness and meaning varies by modifier type
# - m3 appears to be the most parsimonious model that captures the key effects


m2_coef = summary(m2)$coefficients
m3_coef = summary(m3)$coefficients


# Create mediation effects dataframe more simply
mediation_effects <- data.frame(
  modifier = c("kind of", "quite", "slightly", "very"),
  original_effect = m2_coef[grep("countryUS:modifier", rownames(m2_coef)), "Estimate"],
  original_p = m2_coef[grep("countryUS:modifier", rownames(m2_coef)), "Pr(>|t|)"],
  controlled_effect = m3_coef[grep("countryUS:modifier", rownames(m3_coef)), "Estimate"],
  controlled_p = m3_coef[grep("countryUS:modifier", rownames(m3_coef)), "Pr(>|t|)"]
) %>%
  mutate(
    reduction = abs(original_effect) - abs(controlled_effect),
    percent_reduction = (reduction / abs(original_effect)) * 100,
    significance = case_when(
      original_p < 0.01 ~ "**",
      original_p < 0.05 ~ "*",
      original_p < 0.1 ~ "+",
      TRUE ~ "ns"
    )
  )

# Plot mediation effects
ggplot(mediation_effects, aes(x = reorder(modifier, reduction))) +
  geom_segment(aes(y = 0, yend = reduction, xend = modifier), 
              color = "gray50") +
  geom_point(aes(y = reduction), color = "coral", size = 3) +
  geom_text(aes(y = reduction, 
                label = paste0(significance, "\n(", round(original_effect, 3), 
                             " → ", round(controlled_effect, 3), ")")), 
            vjust = -1) +
  theme_minimal() +
  labs(
    title = "Reduction in US-UK Differences After Controlling for Politeness",
    subtitle = "** p<.01, * p<.05, + p<.1, ns = not significant\nValues show coefficients before → after",
    y = "Reduction in |US-UK Difference|",
    x = "Modifier"
  ) +
  coord_flip()


  # Extract fitted values and residuals
fitted_values <- fitted(base_model)
residuals_values <- residuals(base_model)

# Create a data frame for plotting
residuals_df <- data.frame(
  Fitted = fitted_values,
  Residuals = residuals_values
)

# Plot Residuals vs Fitted Values
ggplot(residuals_df, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5) +  # Add points with some transparency
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +  # Add a horizontal line at y=0
  theme_minimal() +
  labs(
    title = "Residuals vs Fitted Values",
    x = "Fitted Values",
    y = "Residuals"
  )

powerSim(base_model, nsim = 100)  # Simulate 100 power estimates

#anova(model , type = "III")