

# ====== Path ======
setwd("/Users/lana/Downloads/paired_data")  # 必要なら変更
# 必要なパッケージの読み込み
library(dplyr)
library(stringr)

# 1. CSVファイルの抽出
csv_files <- list.files(pattern = "^paired_.*\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop("CSVファイルが見つかりませんでした。")
}

# 2. ファイル名の解析と最新バージョンの選択
file_info <- data.frame(
  filepath = csv_files,
  filename = basename(csv_files),
  stringsAsFactors = FALSE
)

# ファイル名から数字を抽出
file_info <- file_info %>%
  mutate(
    numbers = str_extract(filename, "(?<=paired_).*(?=\\.csv)"),
    parts = str_split(numbers, "_")
  ) %>%
  rowwise() %>%
  mutate(
    participant_id = as.integer(parts[1]),
    block_id = as.integer(parts[2]),
    version = if (length(parts) >= 3) as.integer(parts[3]) else 0L
  ) %>%
  ungroup() %>%
  select(-numbers, -parts)

# 各参加者IDとブロックIDの組み合わせで、バージョンが最大のものを選択
selected_files <- file_info %>%
  group_by(participant_id, block_id) %>%
  filter(version == max(version)) %>%
  ungroup()

cat("選択されたファイル:\n")
print(selected_files %>% select(filename, participant_id, block_id, version))

# 3. 選択されたCSVファイルを読み込んで結合
combined_data <- bind_rows(
  lapply(selected_files$filepath, function(file) {
    # CSVを読み込み（空の列を自動削除）
    data <- read.csv(file, stringsAsFactors = FALSE, check.names = TRUE)
    
    # 列名のトリミング（前後の空白を削除）
    colnames(data) <- trimws(colnames(data))
    
    # 空の列を削除（全てがNAまたは空文字の列）
    data <- data %>%
      select(where(~ !all(is.na(.) | . == "")))
    
    # ファイル情報を追加
    file_name <- basename(file)
    data$source_file <- file_name
    
    return(data)
  })
)

# 不要な列名パターンを削除（...で始まる列名）
combined_data <- combined_data %>%
  select(-starts_with("..."))

# speed列を追加（blockNo列が存在する場合）
if ("blockNo" %in% colnames(combined_data)) {
  combined_data <- combined_data %>%
    mutate(speed = ifelse(blockNo %% 2 == 1, 6, 2))
  cat("speed列を追加しました（奇数ブロック=6, 偶数ブロック=2）\n")
} else {
  warning("blockNo列が見つかりませんでした。speed列は追加されませんでした。")
}

# 4. 結合されたデータを保存
output_file <- "combined_paired_data.csv"
write.csv(combined_data, output_file, row.names = FALSE)

cat(sprintf("\n結合完了: %d個のファイルを結合しました。\n", nrow(selected_files)))
cat(sprintf("出力ファイル: %s\n", output_file))
cat(sprintf("総行数: %d行\n", nrow(combined_data)))
cat(sprintf("列数: %d列\n", ncol(combined_data)))