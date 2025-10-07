# ====== Libraries ======
library(tidyverse)

# ====== Path ======
data_dir <- "/Users/lana/Downloads/Overcoherency_data"  # 必要なら変更

# ====== Collect target files (.csv only; exclude .csv.csv) ======
files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
files <- files[!grepl("\\.csv\\.csv$", files, ignore.case = TRUE)]

# ====== Parse id & block from filename: RDK_<id>_<block>_<date>_<time>.csv ======
parse_meta <- function(path){
  tk <- strsplit(basename(path), "_")[[1]]
  if (length(tk) < 5 || !identical(tk[1], "RDK")) {
    return(tibble(file = path, id = NA_character_, block = NA_integer_))
  }
  tibble(
    file  = path,
    id    = tk[2],
    block = suppressWarnings(as.integer(tk[3]))
  )
}

meta <- map_dfr(files, parse_meta) %>%
  filter(!is.na(id), !is.na(block), block >= 1, block <= 20) %>%
  arrange(id, block, file)

message("検出ID: ", paste(unique(meta$id), collapse = ", "))

# ====== Per-participant merge & write ======
all_merged_list <- list()

for (pid in setdiff(unique(meta$id), "5")) {
#for (pid in unique(meta$id)) {
  df <- meta %>% filter(id == pid)
  
  # 1) 各ファイルは "全列 character" のまま読み込む（ここでは型変換しない）
  dfs_char <- purrr::map(df$file, ~{
    readr::read_csv(.x, col_types = readr::cols(.default = readr::col_character()),
                    show_col_types = FALSE)
  })
  
  # 2) まず character のまま結合（ここで型不一致は起きない）
  merged_i <- dplyr::bind_rows(dfs_char, .id = "source_index")
  
  # 3) 一度だけ型推定（数値にできる列を数値化）
  merged_i <- readr::type_convert(merged_i, na = c("", "NA"))
  
  # 4) 90 <-> 270 の入れ替え（数値前提で安全に）
  merged_i <- merged_i %>%
    mutate(
      response_deg   = suppressWarnings(as.numeric(response_deg)),
      direction_cond = suppressWarnings(as.numeric(direction_cond)),
      direction_cond = dplyr::case_when(
        direction_cond == 90  ~ 270,
        direction_cond == 270 ~ 90,
        TRUE ~ direction_cond
      ),
      # 5) 正誤上書き（どちらか NA なら NA）
      correct = dplyr::if_else(direction_cond == response_deg, 1L, 0L,
                               missing = NA_integer_)
    )
  
  # 6) 書き出し
  out_path <- file.path(data_dir, paste0("RDK_", pid, "_merged.csv"))
  readr::write_csv(merged_i, out_path)
  message("Saved: ", out_path)
  
  all_merged_list[[pid]] <- merged_i
}

# ====== All participants merged (robust) ======
# いったん全列 character にしてから結合 → 一括で型推定
all_merged_raw <- dplyr::bind_rows(
  lapply(all_merged_list, function(x) dplyr::mutate(x, dplyr::across(everything(), as.character)))
)
all_merged <- readr::type_convert(all_merged_raw, na = c("", "NA"))

readr::write_csv(all_merged, file.path(data_dir, "RDK_all_merged.csv"))
message("Saved: ", file.path(data_dir, "RDK_all_merged.csv"))
