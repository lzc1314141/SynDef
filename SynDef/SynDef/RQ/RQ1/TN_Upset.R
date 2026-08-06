
library(UpSetR)
library(ComplexUpset)
library(ggplot2)


setwd("D:/cursor code/SynDef/Upset")


set_files <- c(
  "GLANCE-LR"           = "glancelr-TN.csv",
  "N-gram"              = "ngram-TN.csv",
  "PMD"                 = "PMD-TN.csv",
  "LineDP"              = "linedp-TN.csv",
  "DeepLineDP"          = "deeplinedp-TN.csv",
  "SynDef"  = "SynDef-TN.csv",
  "LineDef"             = "linedef-TN.csv",
  "SPLICE-F"            = "splice-F-TN.csv"
)


read_set_ids <- function(csv_file) {
  if (!file.exists(csv_file)) {
    warning(paste("文件不存在:", csv_file))
    return(character(0))
  }
  df <- read.csv(csv_file, header = FALSE, stringsAsFactors = FALSE)
  if (ncol(df) < 3) {
    warning(paste("文件列数不足 3 列，跳过:", csv_file))
    return(character(0))
  }
  paste(df[[1]], df[[2]], df[[3]], sep = "|")
}


set_list <- lapply(set_files, read_set_ids)


if (all(vapply(set_list, length, integer(1)) == 0L)) {
  stop("所有 CSV 读入集合均为空，请检查文件路径或内容。")
}


upset_ready <- lapply(set_list, unique)
upset_data <- UpSetR::fromList(upset_ready)
upset_df <- as.data.frame(upset_data)

png(
  filename = "TN_upset_8sets.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df,
  intersect = names(upset_ready),
  name = "Intersection Size",
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

png(
  filename = "TN_upset_8sets_top30.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df,
  intersect = names(upset_ready),
  name = "Intersection Size",
  n_intersections = 30,
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

png(
  filename = "TN_upset_8sets_top40.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df,
  intersect = names(upset_ready),
  name = "Intersection Size",
  n_intersections = 40,
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

png(
  filename = "TN_upset_8sets_top50.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df,
  intersect = names(upset_ready),
  name = "Intersection Size",
  n_intersections = 50,
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

set_files_6 <- c(
  "GLANCE-LR"          = "glancelr-TN.csv",
  "DeepLineDP"          = "deeplinedp-TN.csv",
  "SPLICE-F"           = "splice-F-TN.csv",
  "PMD"                = "PMD-TN.csv",
  "LineDef"            = "linedef-TN.csv",
  "SynDef" = "SynDef-TN.csv"
)

set_list_6 <- lapply(set_files_6, read_set_ids)
upset_ready_6 <- lapply(set_list_6, unique)
upset_data_6 <- UpSetR::fromList(upset_ready_6)
upset_df_6 <- as.data.frame(upset_data_6)


png(
  filename = "TN_upset_6sets.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df_6,
  intersect = names(upset_ready_6),
  name = "Intersection Size",
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

png(
  filename = "TN_upset_6sets_top30.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df_6,
  intersect = names(upset_ready_6),
  name = "Intersection Size",
  n_intersections = 30,
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

png(
  filename = "TN_upset_6sets_top40.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df_6,
  intersect = names(upset_ready_6),
  name = "Intersection Size",
  n_intersections = 40,
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

png(
  filename = "TN_upset_6sets_top50.png",
  width = 10,
  height = 7,
  units = "in",
  res = 300
)

ComplexUpset::upset(
  upset_df_6,
  intersect = names(upset_ready_6),
  name = "Intersection Size",
  n_intersections = 50,
  base_annotations = list(
    "Intersection Size" = intersection_size(
      text = list(
        size  = 2.5,
        angle = 90,
        vjust = 0.5,
        hjust = -0.2
      )
    )
  ),
  set_sizes = FALSE,
  sort_sets = FALSE,
  sort_intersections_by = "cardinality"
) +
  theme(
    text = element_text(size = 12),
    plot.margin = margin(5, 5, 5, 5)
  )

dev.off()

if (!interactive()) {
  try(browseURL(normalizePath("TN_upset_8sets_top30.png")), silent = TRUE)
}
