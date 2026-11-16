# 13_propaganda_pilot_analysis.R
# Empirical validation: TEE variance decomposition on propaganda/LLM pilot data.
# Compares LLM judge disagreement to human rater disagreement (9 RAs).
# Tests whether TEE-optimal LLM configs produce labels closer to human consensus.
#
# Usage:
#   Rscript analysis/13_propaganda_pilot_analysis.R

library(tidyverse)
library(jsonlite)
library(lme4)
library(patchwork)

set.seed(42)

cat("=== 13_propaganda_pilot_analysis.R ===\n")
cat("Propaganda pilot: TEE decomposition + human comparison\n\n")

fig_dir <- "figures"
out_dir <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- Theme (consistent with 03_figures.R) ---
theme_tle <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 9),
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10)
  )


# =========================================================================
# Step 1: Clean and merge
# =========================================================================
cat("--- Step 1: Clean and merge ---\n")

raw <- stream_in(file("data/raw/propaganda_pilot.jsonl"), verbose = FALSE)

cat("  Raw rows:", nrow(raw), "\n")
cat("  NAs in cn_more_favorable:", sum(is.na(raw$cn_more_favorable)), "\n")
cat("  Errors:", sum(!is.na(raw$error)), "\n")

# Load pilot items (for question_type, question_focus, disagree_bin)
pilot_items <- read_csv("data/processed/propaganda_pilot_items.csv",
                         show_col_types = FALSE)

# Load human disagreement data
human <- read_csv("data/processed/human_tee_item_disagreement.csv",
                   show_col_types = FALSE)

# Clean: drop NAs, merge metadata
df <- raw %>%
  filter(!is.na(cn_more_favorable)) %>%
  select(question_ID, variant_id, judge_model, temperature, replication,
         presentation_order, cn_more_favorable) %>%
  left_join(
    pilot_items %>% select(question_ID, question_type, question_focus,
                           mean_favorability, sd_favorability, disagree_bin),
    by = "question_ID"
  ) %>%
  mutate(
    question_ID  = as.factor(question_ID),
    variant_id   = as.factor(variant_id),
    judge_model  = as.factor(judge_model),
    temperature  = as.factor(temperature),
    question_type = as.factor(question_type),
    disagree_bin = factor(disagree_bin, levels = c("low", "medium", "high"))
  )

cat("  Clean rows:", nrow(df), "(dropped", nrow(raw) - nrow(df), "NAs)\n")
cat("  Items:", n_distinct(df$question_ID), "\n")
cat("  Judges:", levels(df$judge_model), "\n")
cat("  Temps:", levels(df$temperature), "\n")
cat("  Variants:", levels(df$variant_id), "\n")
cat("  Reps per cell:", n_distinct(df$replication), "\n")

# Save cleaned data
write_csv(df, file.path(out_dir, "propaganda_pilot_clean.csv"))
cat("  Saved: data/processed/propaganda_pilot_clean.csv\n\n")


# =========================================================================
# Step 2: TEE variance decomposition
# =========================================================================
cat("--- Step 2: TEE variance decomposition ---\n")

mod <- lmer(
  cn_more_favorable ~ temperature + judge_model +
    (1 | question_type) +
    (1 | question_ID) +
    (1 | variant_id) +
    (1 | question_ID:variant_id) +
    (1 | question_ID:temperature) +
    (1 | variant_id:temperature) +
    (1 | question_ID:judge_model) +
    (1 | variant_id:judge_model),
  data = df,
  REML = TRUE,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)

cat("\nModel summary:\n")
print(summary(mod))

# Check convergence
if (length(mod@optinfo$conv$lme4$messages) > 0) {
  cat("\nWarnings:", paste(mod@optinfo$conv$lme4$messages, collapse = "; "), "\n")
}

# Extract variance components
vc <- as.data.frame(VarCorr(mod))

tle_labels <- c(
  "question_type"              = "between-category",
  "question_ID"                = "within-category item",
  "variant_id"                 = "prompt",
  "question_ID:variant_id"     = "item x prompt",
  "question_ID:temperature"    = "item x temperature",
  "variant_id:temperature"     = "prompt x temperature",
  "question_ID:judge_model"    = "item x judge",
  "variant_id:judge_model"     = "prompt x judge",
  "Residual"                   = "generation"
)

vc_clean <- vc %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor) %>%
  mutate(
    tle_label = tle_labels[component],
    tle_label = ifelse(is.na(tle_label), component, tle_label)
  )

# Temperature fixed effect sensitivity index
temp_contrasts <- fixef(mod)[grepl("temperature", names(fixef(mod)))]
grand_mean <- fixef(mod)["(Intercept)"]
temp_cell_means <- c(grand_mean, grand_mean + temp_contrasts)
sigma2_temp <- sum((temp_cell_means - mean(temp_cell_means))^2) / length(temp_cell_means)

vc_clean <- bind_rows(
  vc_clean,
  tibble(
    component = "temperature (fixed)",
    variance = sigma2_temp,
    sd = sqrt(sigma2_temp),
    tle_label = "temperature (design sensitivity)"
  )
)

# Judge model fixed effect sensitivity index
judge_contrasts <- fixef(mod)[grepl("judge_model", names(fixef(mod)))]
judge_cell_means <- c(grand_mean, grand_mean + judge_contrasts)
sigma2_judge <- sum((judge_cell_means - mean(judge_cell_means))^2) / length(judge_cell_means)

vc_clean <- bind_rows(
  vc_clean,
  tibble(
    component = "judge_model (fixed)",
    variance = sigma2_judge,
    sd = sqrt(sigma2_judge),
    tle_label = "judge model (design sensitivity)"
  )
)

# Clamp negative REML estimates before computing proportions
vc_clean <- vc_clean %>%
  mutate(variance = pmax(variance, 0))
total_var <- sum(vc_clean$variance)
vc_clean <- vc_clean %>%
  mutate(
    pct_total = round(100 * variance / total_var, 2),
    tier = case_when(
      tle_label %in% c("generation", "prompt", "item x prompt",
                        "prompt x temperature", "prompt x judge") ~ "Tier 1",
      tle_label %in% c("between-category", "within-category item",
                        "item x temperature", "item x judge") ~ "Signal / Item",
      TRUE ~ "Tier 2"
    )
  ) %>%
  arrange(desc(variance))

cat("\n=== TEE Variance Components (Propaganda Pilot) ===\n")
vc_print <- vc_clean %>% select(tle_label, variance, pct_total, tier)
print(as.data.frame(vc_print))

write_csv(vc_clean, file.path(out_dir, "propaganda_pilot_variance_components.csv"))
cat("\nSaved: data/processed/propaganda_pilot_variance_components.csv\n\n")


# =========================================================================
# Step 3: Human vs LLM disagreement at item level
# =========================================================================
cat("--- Step 3: Human vs LLM disagreement ---\n")

# Human disagreement: SD of binary votes (from human data)
human_disagree <- human %>%
  filter(item_id %in% levels(df$question_ID)) %>%
  select(item_id, sd_fav, mean_fav, consensus) %>%
  rename(question_ID = item_id,
         human_disagree = sd_fav,
         human_mean = mean_fav)

# LLM disagreement: SD of binary votes across all conditions
llm_item <- df %>%
  group_by(question_ID) %>%
  summarise(
    llm_mean = mean(cn_more_favorable),
    llm_disagree = sd(cn_more_favorable),
    n_obs = n(),
    .groups = "drop"
  )

# Item BLUPs from lmer
item_blups <- ranef(mod)$question_ID %>%
  rownames_to_column("question_ID") %>%
  rename(item_blup = `(Intercept)`)

# Merge everything
item_compare <- llm_item %>%
  left_join(human_disagree, by = "question_ID") %>%
  left_join(item_blups, by = "question_ID") %>%
  left_join(
    pilot_items %>% select(question_ID, question_type, question_focus, disagree_bin),
    by = "question_ID"
  )

cat("  Items with both human and LLM data:", nrow(item_compare), "\n")

# Correlations
cor_disagree <- cor(item_compare$human_disagree, item_compare$llm_disagree,
                     use = "complete.obs")
cor_mean <- cor(item_compare$human_mean, item_compare$llm_mean,
                 use = "complete.obs")
cor_test_disagree <- cor.test(item_compare$human_disagree, item_compare$llm_disagree)
cor_test_mean <- cor.test(item_compare$human_mean, item_compare$llm_mean)

cat("\n  Correlation (human vs LLM disagreement SD):",
    sprintf("r = %.3f [%.3f, %.3f], p = %.4f",
            cor_disagree,
            cor_test_disagree$conf.int[1],
            cor_test_disagree$conf.int[2],
            cor_test_disagree$p.value), "\n")
cat("  Correlation (human vs LLM mean favorability):",
    sprintf("r = %.3f [%.3f, %.3f], p = %.4f",
            cor_mean,
            cor_test_mean$conf.int[1],
            cor_test_mean$conf.int[2],
            cor_test_mean$p.value), "\n")

write_csv(item_compare, file.path(out_dir, "propaganda_pilot_item_comparison.csv"))
cat("  Saved: data/processed/propaganda_pilot_item_comparison.csv\n\n")

# --- Figure: Human vs LLM disagreement scatter ---
p_disagree <- ggplot(item_compare,
       aes(x = human_disagree, y = llm_disagree, color = question_type)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "gray40", linewidth = 0.6) +
  scale_color_brewer(palette = "Set1", name = "Question type") +
  labs(
    title = "Human vs LLM disagreement by item",
    subtitle = sprintf("r = %.3f, 95%% CI [%.3f, %.3f], n = %d items",
                        cor_disagree,
                        cor_test_disagree$conf.int[1],
                        cor_test_disagree$conf.int[2],
                        nrow(item_compare)),
    x = "Human rater disagreement (SD of 9 binary votes)",
    y = "LLM disagreement (SD across all pipeline configs)"
  ) +
  theme_tle

ggsave(file.path(fig_dir, "propaganda_human_vs_llm_disagreement.pdf"),
       p_disagree, width = 7, height = 5)
cat("  Saved: figures/propaganda_human_vs_llm_disagreement.pdf\n")

# --- Figure: Human vs LLM mean favorability ---
p_mean <- ggplot(item_compare,
       aes(x = human_mean, y = llm_mean, color = question_type)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  geom_smooth(method = "lm", se = TRUE, color = "gray40", linewidth = 0.6) +
  scale_color_brewer(palette = "Set1", name = "Question type") +
  labs(
    title = "Human vs LLM mean favorability",
    subtitle = sprintf("r = %.3f; dashed = perfect agreement", cor_mean),
    x = "Human: P(Chinese more favorable), 9 RAs",
    y = "LLM: P(Chinese more favorable), all configs"
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_tle

ggsave(file.path(fig_dir, "propaganda_human_vs_llm_mean.pdf"),
       p_mean, width = 7, height = 5)
cat("  Saved: figures/propaganda_human_vs_llm_mean.pdf\n\n")


# =========================================================================
# Step 4: Accuracy against human consensus
# =========================================================================
cat("--- Step 4: Accuracy against human consensus ---\n")

# Human ground truth: majority vote of 9 RAs
# mean_fav > 0.5 → Chinese favored (coded 1), else English favored (coded 0)
# Ties (mean_fav == 0.5): exclude or assign based on consensus column
human_gt <- human_disagree %>%
  mutate(
    human_gt = case_when(
      human_mean > 0.5 ~ 1L,
      human_mean < 0.5 ~ 0L,
      # For ties, use consensus column
      consensus == "Chinese favored" ~ 1L,
      TRUE ~ 0L
    )
  )

cat("  Human ground truth: ", sum(human_gt$human_gt == 1), "Chinese favored,",
    sum(human_gt$human_gt == 0), "English favored\n")

# --- Per-config accuracy (5 variants x 3 judges x 3 temps = 45 configs) ---
config_acc <- df %>%
  group_by(variant_id, judge_model, temperature, question_ID) %>%
  summarise(
    llm_vote = as.integer(mean(cn_more_favorable) > 0.5),  # majority of 5 reps
    .groups = "drop"
  ) %>%
  left_join(human_gt %>% select(question_ID, human_gt), by = "question_ID") %>%
  group_by(variant_id, judge_model, temperature) %>%
  summarise(
    accuracy = mean(llm_vote == human_gt, na.rm = TRUE),
    n_items = n(),
    .groups = "drop"
  )

cat("\n  Per-config accuracy summary:\n")
cat("    Mean:", sprintf("%.1f%%", 100 * mean(config_acc$accuracy)), "\n")
cat("    Min: ", sprintf("%.1f%%", 100 * min(config_acc$accuracy)),
    " | Max:", sprintf("%.1f%%", 100 * max(config_acc$accuracy)), "\n")
cat("    SD:  ", sprintf("%.1f%%", 100 * sd(config_acc$accuracy)), "\n\n")

# --- TEE-optimal: average across ALL conditions, then majority vote ---
tee_optimal <- df %>%
  group_by(question_ID) %>%
  summarise(
    llm_mean = mean(cn_more_favorable),
    llm_vote = as.integer(mean(cn_more_favorable) > 0.5),
    n_obs = n(),
    .groups = "drop"
  ) %>%
  left_join(human_gt %>% select(question_ID, human_gt), by = "question_ID")

tee_accuracy <- mean(tee_optimal$llm_vote == tee_optimal$human_gt, na.rm = TRUE)
cat("  TEE-optimal accuracy (average over all configs):",
    sprintf("%.1f%%", 100 * tee_accuracy), "\n")

# --- Naive: single random prompt, single judge, single temp ---
# Simulate 1000 naive draws
set.seed(42)
naive_draws <- replicate(1000, {
  v <- sample(levels(df$variant_id), 1)
  j <- sample(levels(df$judge_model), 1)
  t <- sample(levels(df$temperature), 1)

  naive_df <- df %>%
    filter(variant_id == v, judge_model == j, temperature == t) %>%
    group_by(question_ID) %>%
    summarise(
      llm_vote = as.integer(mean(cn_more_favorable) > 0.5),
      .groups = "drop"
    ) %>%
    left_join(human_gt %>% select(question_ID, human_gt), by = "question_ID")

  mean(naive_df$llm_vote == naive_df$human_gt, na.rm = TRUE)
})

naive_mean <- mean(naive_draws)
naive_ci <- quantile(naive_draws, c(0.025, 0.975))
cat("  Naive accuracy (random single config):",
    sprintf("%.1f%% [%.1f%%, %.1f%%]",
            100 * naive_mean, 100 * naive_ci[1], 100 * naive_ci[2]), "\n")
cat("  TEE advantage:", sprintf("%.1f pp", 100 * (tee_accuracy - naive_mean)), "\n\n")

# --- Accuracy by judge model ---
judge_acc <- config_acc %>%
  group_by(judge_model) %>%
  summarise(
    mean_acc = mean(accuracy),
    sd_acc = sd(accuracy),
    .groups = "drop"
  )
cat("  Accuracy by judge model:\n")
for (i in 1:nrow(judge_acc)) {
  cat(sprintf("    %s: %.1f%% (SD %.1f%%)\n",
              judge_acc$judge_model[i],
              100 * judge_acc$mean_acc[i],
              100 * judge_acc$sd_acc[i]))
}

# --- Accuracy by temperature ---
temp_acc <- config_acc %>%
  group_by(temperature) %>%
  summarise(
    mean_acc = mean(accuracy),
    sd_acc = sd(accuracy),
    .groups = "drop"
  )
cat("\n  Accuracy by temperature:\n")
for (i in 1:nrow(temp_acc)) {
  cat(sprintf("    T=%s: %.1f%% (SD %.1f%%)\n",
              temp_acc$temperature[i],
              100 * temp_acc$mean_acc[i],
              100 * temp_acc$sd_acc[i]))
}

# Save config accuracy
write_csv(config_acc, file.path(out_dir, "propaganda_pilot_config_accuracy.csv"))
cat("\n  Saved: data/processed/propaganda_pilot_config_accuracy.csv\n\n")


# =========================================================================
# Step 5: TEE design recommendations
# =========================================================================
cat("--- Step 5: TEE design recommendations ---\n\n")

# Identify dominant components
top3 <- vc_clean %>%
  head(3) %>%
  select(tle_label, pct_total, tier)
cat("  Top 3 variance components:\n")
for (i in 1:3) {
  cat(sprintf("    %d. %s: %.1f%% (%s)\n",
              i, top3$tle_label[i], top3$pct_total[i], top3$tier[i]))
}

# --- Compare: multiple judges vs single judge ---
# "TEE recommendation": average across all 3 judges (for each prompt x temp)
multi_judge <- df %>%
  group_by(variant_id, temperature, question_ID) %>%
  summarise(
    llm_vote = as.integer(mean(cn_more_favorable) > 0.5),
    .groups = "drop"
  ) %>%
  left_join(human_gt %>% select(question_ID, human_gt), by = "question_ID") %>%
  group_by(variant_id, temperature) %>%
  summarise(accuracy = mean(llm_vote == human_gt, na.rm = TRUE), .groups = "drop")

# Single judge (each config is one judge)
single_judge <- config_acc

cat("\n  Multi-judge accuracy (averaged over 3 judges per prompt x temp):\n")
cat(sprintf("    Mean: %.1f%% (SD %.1f%%)\n",
            100 * mean(multi_judge$accuracy),
            100 * sd(multi_judge$accuracy)))
cat(sprintf("  Single-judge accuracy:\n"))
cat(sprintf("    Mean: %.1f%% (SD %.1f%%)\n",
            100 * mean(single_judge$accuracy),
            100 * sd(single_judge$accuracy)))
cat(sprintf("  Multi-judge advantage: %.1f pp\n",
            100 * (mean(multi_judge$accuracy) - mean(single_judge$accuracy))))

# --- Compare: multiple prompts vs single prompt ---
multi_prompt <- df %>%
  group_by(judge_model, temperature, question_ID) %>%
  summarise(
    llm_vote = as.integer(mean(cn_more_favorable) > 0.5),
    .groups = "drop"
  ) %>%
  left_join(human_gt %>% select(question_ID, human_gt), by = "question_ID") %>%
  group_by(judge_model, temperature) %>%
  summarise(accuracy = mean(llm_vote == human_gt, na.rm = TRUE), .groups = "drop")

cat(sprintf("\n  Multi-prompt accuracy (averaged over 5 prompts per judge x temp):\n"))
cat(sprintf("    Mean: %.1f%% (SD %.1f%%)\n",
            100 * mean(multi_prompt$accuracy),
            100 * sd(multi_prompt$accuracy)))
cat(sprintf("  Multi-prompt advantage over single-config: %.1f pp\n",
            100 * (mean(multi_prompt$accuracy) - mean(single_judge$accuracy))))

# --- Full TEE vs worst single config ---
worst_config <- config_acc %>% slice_min(accuracy, n = 1)
best_config  <- config_acc %>% slice_max(accuracy, n = 1)
cat(sprintf("\n  Best single config:  %.1f%% (variant=%s, judge=%s, temp=%s)\n",
            100 * best_config$accuracy[1],
            best_config$variant_id[1],
            best_config$judge_model[1],
            best_config$temperature[1]))
cat(sprintf("  Worst single config: %.1f%% (variant=%s, judge=%s, temp=%s)\n",
            100 * worst_config$accuracy[1],
            worst_config$variant_id[1],
            worst_config$judge_model[1],
            worst_config$temperature[1]))
cat(sprintf("  TEE-optimal:         %.1f%%\n", 100 * tee_accuracy))
cat(sprintf("  Config range:        %.1f pp\n",
            100 * (best_config$accuracy[1] - worst_config$accuracy[1])))

# --- Accuracy by human disagreement stratum ---
strat_acc <- tee_optimal %>%
  left_join(pilot_items %>% select(question_ID, disagree_bin), by = "question_ID") %>%
  group_by(disagree_bin) %>%
  summarise(
    accuracy = mean(llm_vote == human_gt, na.rm = TRUE),
    n_items = n(),
    .groups = "drop"
  )

cat("\n  TEE-optimal accuracy by human disagreement stratum:\n")
for (i in 1:nrow(strat_acc)) {
  cat(sprintf("    %s: %.1f%% (n=%d items)\n",
              strat_acc$disagree_bin[i],
              100 * strat_acc$accuracy[i],
              strat_acc$n_items[i]))
}


# =========================================================================
# Multi-panel summary figure
# =========================================================================
cat("\n--- Generating multi-panel figure ---\n")

# Panel A: Variance decomposition forest plot
vc_plot <- vc_clean %>%
  filter(variance > 0) %>%
  mutate(tle_label = fct_reorder(tle_label, variance))

p_forest <- ggplot(vc_plot, aes(x = variance, y = tle_label, fill = tier)) +
  geom_col(alpha = 0.85) +
  geom_text(aes(label = sprintf("%.1f%%", pct_total)),
            hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("Tier 1" = "#2166ac",
                                "Tier 2" = "#b2182b",
                                "Signal / Item" = "#4daf4a"),
                     name = "Component type") +
  labs(title = "A. TEE variance decomposition",
       subtitle = "Propaganda pilot (50 items, 3 judges, 5 prompts, 3 temps)",
       x = "Variance", y = NULL) +
  theme_tle +
  theme(legend.position = "right")

# Panel B: Human vs LLM disagreement
p_b <- p_disagree +
  labs(title = "B. Human vs LLM disagreement")

# Panel C: Accuracy by config (dot plot)
p_config <- ggplot(config_acc,
                    aes(x = accuracy, y = judge_model, color = temperature)) +
  geom_jitter(height = 0.15, size = 2, alpha = 0.7) +
  geom_vline(xintercept = tee_accuracy, linetype = "dashed", color = "black") +
  annotate("text", x = tee_accuracy + 0.005, y = 3.4,
           label = sprintf("TEE optimal\n%.1f%%", 100 * tee_accuracy),
           hjust = 0, size = 3, color = "black") +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_color_brewer(palette = "YlOrRd", name = "Temperature") +
  labs(title = "C. Accuracy vs human consensus by config",
       subtitle = sprintf("45 configs; naive mean = %.1f%%", 100 * naive_mean),
       x = "Accuracy (fraction matching human majority)", y = NULL) +
  theme_tle

# Panel D: Accuracy by disagreement stratum
strat_long <- tee_optimal %>%
  left_join(pilot_items %>% select(question_ID, disagree_bin), by = "question_ID") %>%
  mutate(correct = as.integer(llm_vote == human_gt))

# Compute stratum-level stats for error bars (avoid Hmisc dependency)
strat_stats <- strat_long %>%
  group_by(disagree_bin) %>%
  summarise(
    mean_acc = mean(correct, na.rm = TRUE),
    se = sqrt(mean_acc * (1 - mean_acc) / n()),
    lo = pmax(0, mean_acc - 1.96 * se),
    hi = pmin(1, mean_acc + 1.96 * se),
    .groups = "drop"
  )

p_strat <- ggplot(strat_stats, aes(x = disagree_bin, y = mean_acc, fill = disagree_bin)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2) +
  scale_fill_manual(values = c("low" = "#66c2a5", "medium" = "#fc8d62",
                                "high" = "#8da0cb"),
                     name = "Human disagreement") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  labs(title = "D. TEE accuracy by human disagreement stratum",
       subtitle = "Higher human disagreement = harder for LLM",
       x = "Human disagreement stratum", y = "Accuracy") +
  theme_tle +
  theme(legend.position = "none")

# Combine
p_combined <- (p_forest | p_b) / (p_config | p_strat) +
  plot_annotation(
    title = "TEE Variance Decomposition: Propaganda/LLM Pilot",
    subtitle = "50 items, 3 LLM judges, 5 prompt variants, 3 temperatures, 5 replications",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(color = "gray40", size = 10)
    )
  )

ggsave(file.path(fig_dir, "propaganda_pilot_tee_summary.pdf"),
       p_combined, width = 16, height = 12)
ggsave(file.path(fig_dir, "propaganda_pilot_tee_summary.png"),
       p_combined, width = 16, height = 12, dpi = 150)
cat("  Saved: figures/propaganda_pilot_tee_summary.pdf\n")
cat("  Saved: figures/propaganda_pilot_tee_summary.png\n")

# Save individual panels too
ggsave(file.path(fig_dir, "propaganda_pilot_forest.pdf"),
       p_forest, width = 8, height = 5)

cat("\n=== Done ===\n")
