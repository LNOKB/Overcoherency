# 必要なパッケージの読み込み
library(dplyr)
library(ggplot2)

# データの読み込みpaired_2_20
data <- read.csv("combined_paired_data.csv", stringsAsFactors = FALSE)

# outframeの条件による割合を計算
total_rows <- nrow(data)
outframe1_only <- sum(data$outframe1 < 7 & data$outframe2 >= 7, na.rm = TRUE)
outframe2_only <- sum(data$outframe2 < 7 & data$outframe1 >= 7, na.rm = TRUE)
both_outframes <- sum(data$outframe1 < 7 & data$outframe2 < 7, na.rm = TRUE)

cat("=== Outframe条件の割合 ===\n")
cat(sprintf("outframe1のみ < 7: %.2f%% (%d/%d)\n", 
            (outframe1_only / total_rows) * 100, outframe1_only, total_rows))
cat(sprintf("outframe2のみ < 7: %.2f%% (%d/%d)\n", 
            (outframe2_only / total_rows) * 100, outframe2_only, total_rows))
cat(sprintf("両方とも < 7: %.2f%% (%d/%d)\n", 
            (both_outframes / total_rows) * 100, both_outframes, total_rows))
cat(sprintf("合計（少なくとも1つ < 7）: %.2f%%\n\n", 
            ((outframe1_only + outframe2_only + both_outframes) / total_rows) * 100))

# データのフィルタリング
filtered_data <- data %>%
  filter(outframe1 < 7, outframe2 < 7) %>%
  filter(participantNo == 2)
  
cat(sprintf("フィルタリング後のデータ数: %d行\n\n", nrow(filtered_data)))

# resp_center_is_more_coherentの回答率を計算
# TRUEまたは1を「正答」として扱う
plot_data_center <- filtered_data %>%
  mutate(
    # resp_center_is_more_coherentを数値化（TRUE/1 = 1, FALSE/0 = 0）
    response = as.numeric(resp_center_is_more_coherent)
  ) %>%
  group_by(centralCoh, peripheralCoh, speed) %>%
  summarise(
    response_rate = mean(response, na.rm = TRUE) * 100,
    n = n(),
    .groups = "drop"
  )

# resp_dir_TorFの回答率を計算
plot_data_dir <- filtered_data %>%
  mutate(
    # resp_dir_TorFを数値化（TRUE/1 = 1, FALSE/0 = 0）
    response = as.numeric(resp_dir_TorF)
  ) %>%
  group_by(centralCoh, peripheralCoh, speed) %>%
  summarise(
    response_rate = mean(response, na.rm = TRUE) * 100,
    n = n(),
    .groups = "drop"
  )

# speedをfactorに変換（凡例用）
plot_data_center$speed <- factor(plot_data_center$speed, 
                                 levels = c(2, 6),
                                 labels = c("2", "6"))

plot_data_dir$speed <- factor(plot_data_dir$speed, 
                              levels = c(2, 6),
                              labels = c("2", "6"))

# APAスタイルのテーマを作成
apa_theme <- theme_classic(base_size = 12, base_family = "sans") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 11, color = "black"),
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10),
    legend.position = "right",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    strip.background = element_rect(fill = "white", color = "black", linewidth = 0.5),
    strip.text = element_text(face = "bold", size = 11)
  )

# グラフ1: resp_center_is_more_coherentの作成（centralCohごとにファセット）
p1 <- ggplot(plot_data_center, aes(x = peripheralCoh, y = response_rate, 
                                   color = speed, group = speed)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~ centralCoh, nrow = 1, 
             labeller = labeller(centralCoh = function(x) paste("Central Coherence =", x))) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_color_manual(values = c("2" = "#2166AC", "6" = "#B2182B"),
                     name = "Speed (deg/sec)") +
  labs(
    x = "Peripheral Coherence",
    y = "'Central was more coherent' Response Rate (%)"
  ) +
  apa_theme

# グラフ2: resp_dir_TorFの作成（centralCohごとにファセット）
p2 <- ggplot(plot_data_dir, aes(x = peripheralCoh, y = response_rate, 
                                color = speed, group = speed)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~ centralCoh, nrow = 1, 
             labeller = labeller(centralCoh = function(x) paste("Central Coherence =", x))) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_color_manual(values = c("2" = "#2166AC", "6" = "#B2182B"),
                     name = "Speed (deg/sec)") +
  labs(
    x = "Peripheral Coherence",
    y = "Direction Judgement Accuracy (%)"
  ) +
  apa_theme

# グラフを表示
print(p1)
print(p2)

# グラフを保存
ggsave("response_rate_center_coherent.png", plot = p1, 
       width = 12, height = 4, dpi = 300, bg = "white")
ggsave("response_rate_center_coherent.pdf", plot = p1, 
       width = 12, height = 4)

ggsave("response_rate_direction.png", plot = p2, 
       width = 12, height = 4, dpi = 300, bg = "white")
ggsave("response_rate_direction.pdf", plot = p2, 
       width = 12, height = 4)

cat("グラフを保存しました:\n")
cat("- response_rate_center_coherent.png/pdf\n")
cat("- response_rate_direction.png/pdf\n")