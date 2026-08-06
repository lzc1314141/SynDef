library(tidyverse)
library(gridExtra)
library(lattice)
library(ModelMetrics)
library(caret)
library(reshape2)
library(car)
library(carData)
library(pROC)
library(effsize)
library(ScottKnottESD)
library(dplyr)
library(tibble)

# Create timestamped directory
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results.dir = paste0('D:/cursor code/results/SynDef_result', timestamp, '/')

dir.create(file.path(results.dir), recursive = TRUE, showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))]
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL
  
  if(reverse == TRUE)
  { 
    ranking <- (max(sk_esd(df)$group)-sk_esd(df)$group) +1 
  }
  else
  { 
    ranking <- sk_esd(df)$group 
  }
  
  x$rank <- paste("Rank",ranking[as.character(gsub("-", ".", x$variable))])
  return(x)
}

# 读取DeepLineDP结果用于获取ground truth
prediction_dir = 'D:/cursor code/SPLICE-master/Baseline-result/DeepLineDP/output/prediction/DeepLineDP/within-release/'

all_files = list.files(prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}


line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth, is.comment.line)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5 &  is.comment.line== "False")
line.ground.truth = distinct(line.ground.truth)

get.line.metrics.result = function(baseline.df, cur.df.file)
{
  
  baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))
  
  sorted = baseline.df.with.ground.truth %>% group_by(filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())%>% mutate(totalSLOC = n())
  
  IFA = sorted %>%  group_by(filename)  %>% filter(line.level.ground.truth == "True") %>% top_n(1, -order)
  IFA = IFA%>% arrange(filename)
  ifa.list = IFA$order - 1
  
  total_true = sorted %>%  group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  total_true = total_true %>% filter(total_true > 0)
  total_true = total_true%>% arrange(filename)
  
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
  recall20LOC = recall20LOC %>% arrange(filename)
  recall.list = recall20LOC$recall20LOC
  
  effort20Recall = sorted %>% merge(total_true)  %>% group_by(filename) %>% arrange(order, .by_group=TRUE) %>% mutate    (cummulative_correct_pred = cumsum  (line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth ==      "True")/total_true, digits = 2)) %>% mutate(class = case_when((line.level.ground.truth == 'True' & recall <= 0.2) ~ order/n(),TRUE ~ 0))     %>%  summarize(effort20Recall = if_else(max(class)==0, sum(recall <= 0.2)/n(), max(class) ))
  effort20Recall = effort20Recall %>% arrange(filename)
  effort.list = effort20Recall$effort20Recall
  
  fpa = sorted %>% merge(total_true) %>% group_by(filename) %>% arrange(order, .by_group=TRUE) %>% mutate(lineFPA = if_else    (line.level.ground.truth == 'True',  n()-order+1, 0 ) / (n() * total_true)) %>% summarize(FPA = sum(lineFPA) )
  fpa = fpa %>% arrange(filename)
  fpa.list = fpa$FPA
  
  top5 = sorted %>% merge(total_true) %>% group_by(filename) %>% filter(order <= 5)  %>%  summarize(top5 = sum(line.level.ground.truth == "True")/n())
  top5 = top5 %>% arrange(filename)
  top5.list = top5$top5
  
  top10 = sorted %>% merge(total_true)  %>% group_by(filename) %>% filter(order <= 10)  %>%  summarize(top10 = sum(line.level.ground.truth == "True")/n())
  top10 = top10 %>% arrange(filename)
  top10.list = top10$top10
  
  result.df = data.frame(IFA$filename, ifa.list, recall.list, effort.list, fpa.list, top5.list, top10.list, IFA$totalSLOC)
  
  return(result.df)
}

all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

SynDef.result.dir = 'D:/cursor code/SynDef/SynDef_result/'

SynDef.result.df = NULL

## get result from SynDef_result CSV files using existing rank
for(rel in all_eval_releases)
{  
  
  # Read CSV file directly (format: predicted_buggy_lines, predicted_buggy_line_numbers, rank, line_entropy)
  csv_file = paste0(SynDef.result.dir, rel, '-result.csv')
  
  if (!file.exists(csv_file)) {
    warning(paste("File not found:", csv_file))
    next
  }
  
  SynDef.result = read.csv(csv_file)
  
  # Check required columns
  required_cols = c("predicted_buggy_lines", "predicted_buggy_line_numbers", "rank")
  missing_cols = setdiff(required_cols, colnames(SynDef.result))
  if (length(missing_cols) > 0) {
    warning(paste("Missing columns in", csv_file, ":", paste(missing_cols, collapse=", ")))
    next
  }
  
  # Extract file name from predicted_buggy_lines (format: filename:line_number)
  # Format in SynDef_result CSV: path/to/file.java:line_number (already uses / separator)
  SynDef.result$file.name = sapply(strsplit(as.character(SynDef.result$predicted_buggy_lines), ":"), 
                                     function(x) paste(x[-length(x)], collapse=":"))
  
  # Use predicted_buggy_line_numbers as line.number
  SynDef.result$line.number = as.numeric(SynDef.result$predicted_buggy_line_numbers)
  
  # Note: file.name already uses / separator format, no conversion needed
  
  # Recalculate rank per file (since rank in CSV might be global, we need per-file ranking)
  # Sort by rank within each file to maintain the original order
  SynDef.result = SynDef.result %>% 
    group_by(file.name) %>% 
    arrange(rank, .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    ungroup()
  
  SynDef.result = select(SynDef.result, 'file.name', 'line.number', 'rank')
  names(SynDef.result) = c('filename', 'line.number', 'rank')
  
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
  
  print(paste0('正在处理项目: ', rel))
  SynDef.eval.result = get.line.metrics.result(SynDef.result, cur.df.file) %>% mutate(test=rel)
  
  SynDef.result.df = rbind(SynDef.result.df, SynDef.eval.result)
  
  print(paste0('✅ 完成 SynDef_result CSV 处理: ', rel))
  print(paste(rep("=", 50), collapse=""))
}

sum_SynDef.result.df = SynDef.result.df %>% summarize(IFA=median(ifa.list),recall=median(recall.list),effort=median(effort.list), fpa=median(fpa.list), top5=mean(top5.list), top10=mean(top10.list), .by=test)

names(sum_SynDef.result.df) = c("release", "IFA", "Recall20%LOC", "Effort@20%Recall", "FPA", "top5", "top10")

#### get data results of SynDef_result ####
write.csv(sum_SynDef.result.df, file = paste0(results.dir, 'SynDef_result_results.csv'), row.names = FALSE)

# Generate statistics summary
statistics_data <- data.frame(
  Metric = c("IFA", "Recall@20%LOC", "Effort@20%Recall", "FPA", "Top5", "Top10"),
  Mean = c(mean(sum_SynDef.result.df$IFA),
           mean(sum_SynDef.result.df$`Recall20%LOC`),
           mean(sum_SynDef.result.df$`Effort@20%Recall`),
           mean(sum_SynDef.result.df$FPA),
           mean(sum_SynDef.result.df$top5),
           mean(sum_SynDef.result.df$top10)),
  Median = c(median(sum_SynDef.result.df$IFA),
             median(sum_SynDef.result.df$`Recall20%LOC`),
             median(sum_SynDef.result.df$`Effort@20%Recall`),
             median(sum_SynDef.result.df$FPA),
             median(sum_SynDef.result.df$top5),
             median(sum_SynDef.result.df$top10)),
  StdDev = c(sd(sum_SynDef.result.df$IFA),
             sd(sum_SynDef.result.df$`Recall20%LOC`),
             sd(sum_SynDef.result.df$`Effort@20%Recall`),
             sd(sum_SynDef.result.df$FPA),
             sd(sum_SynDef.result.df$top5),
             sd(sum_SynDef.result.df$top10)),
  Min = c(min(sum_SynDef.result.df$IFA),
          min(sum_SynDef.result.df$`Recall20%LOC`),
          min(sum_SynDef.result.df$`Effort@20%Recall`),
          min(sum_SynDef.result.df$FPA),
          min(sum_SynDef.result.df$top5),
          min(sum_SynDef.result.df$top10)),
  Max = c(max(sum_SynDef.result.df$IFA),
          max(sum_SynDef.result.df$`Recall20%LOC`),
          max(sum_SynDef.result.df$`Effort@20%Recall`),
          max(sum_SynDef.result.df$FPA),
          max(sum_SynDef.result.df$top5),
          max(sum_SynDef.result.df$top10))
)

write.csv(statistics_data, file = paste0(results.dir, 'SynDef_result_statistics.csv'), row.names = FALSE)

sum_SynDef.result.df$technique = 'SynDef_result'

recall.result.df = select(sum_SynDef.result.df, c('technique', 'Recall20%LOC'))
ifa.result.df = select(sum_SynDef.result.df, c('technique', 'IFA'))
effort.result.df = select(sum_SynDef.result.df, c('technique', 'Effort@20%Recall'))
fpa.result.df = select(sum_SynDef.result.df, c('technique', 'FPA'))
top5.result.df = select(sum_SynDef.result.df, c('technique', 'top5'))
top10.result.df = select(sum_SynDef.result.df, c('technique', 'top10'))

# Skip ranking for single technique, directly generate plots
colnames(recall.result.df) <- c("technique", "value")
colnames(ifa.result.df) <- c("technique", "value")
colnames(effort.result.df) <- c("technique", "value")
colnames(fpa.result.df) <- c("technique", "value")
colnames(top5.result.df) <- c("technique", "value")
colnames(top10.result.df) <- c("technique", "value")

# Generate individual plots for SynDef with Mean and Median labels
p1 <- ggplot(recall.result.df, aes(x=technique, y=value)) + geom_boxplot(fill="lightblue") +
  stat_summary(fun = mean, geom = "point", shape = 17, size = 1.5, color = "red") + 
  annotate("text", x = 1, y = max(recall.result.df$value) * 0.95, 
           label = paste0("Mean: ", round(mean(recall.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "red") +
  annotate("text", x = 1, y = max(recall.result.df$value) * 0.88, 
           label = paste0("Median: ", round(median(recall.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "blue") +
  ylab("Recall@Top20%LOC") + xlab("") + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
ggsave(paste0(results.dir,"SynDef_result-Recall@Top20LOC.pdf"), width=7, height=2.5)

p2 <- ggplot(effort.result.df, aes(x=technique, y=value)) + geom_boxplot(fill="lightgreen") +
  stat_summary(fun = mean, geom = "point", shape = 17, size = 1.5, color = "red") + 
  annotate("text", x = 1, y = max(effort.result.df$value) * 0.95, 
           label = paste0("Mean: ", round(mean(effort.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "red") +
  annotate("text", x = 1, y = max(effort.result.df$value) * 0.88, 
           label = paste0("Median: ", round(median(effort.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "blue") +
  ylab("Effort@Top20%Recall") + xlab("") + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
ggsave(paste0(results.dir,"SynDef_result-Effort@Top20Recall.pdf"), width=7, height=2.5)

p3 <- ggplot(ifa.result.df, aes(x=technique, y=value)) + geom_boxplot(fill="lightcoral") +
  stat_summary(fun = mean, geom = "point", shape = 17, size = 1.5, color = "red") + 
  annotate("text", x = 1, y = max(ifa.result.df$value) * 0.95, 
           label = paste0("Mean: ", round(mean(ifa.result.df$value), 1)), 
           hjust = 0.5, size = 3, color = "red") +
  annotate("text", x = 1, y = max(ifa.result.df$value) * 0.88, 
           label = paste0("Median: ", round(median(ifa.result.df$value), 1)), 
           hjust = 0.5, size = 3, color = "blue") +
  coord_cartesian(ylim=c(0,max(ifa.result.df$value)*1.1)) + 
  ylab("IFA") + xlab("") + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
ggsave(paste0(results.dir, "SynDef_result-IFA.pdf"), width=7, height=2.5)

p4 <- ggplot(fpa.result.df, aes(x=technique, y=value)) + geom_boxplot(fill="lightyellow") +
  stat_summary(fun = mean, geom = "point", shape = 17, size = 1.5, color = "red") + 
  annotate("text", x = 1, y = max(fpa.result.df$value) * 0.95, 
           label = paste0("Mean: ", round(mean(fpa.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "red") +
  annotate("text", x = 1, y = max(fpa.result.df$value) * 0.88, 
           label = paste0("Median: ", round(median(fpa.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "blue") +
  ylab("FPA") + xlab("") + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
ggsave(paste0(results.dir, "SynDef_result-FPA.pdf"), width=7, height=2.5)

p5 <- ggplot(top5.result.df, aes(x=technique, y=value)) + geom_boxplot(fill="lightpink") +
  stat_summary(fun = mean, geom = "point", shape = 17, size = 1.5, color = "red") + 
  annotate("text", x = 1, y = max(top5.result.df$value) * 0.95, 
           label = paste0("Mean: ", round(mean(top5.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "red") +
  annotate("text", x = 1, y = max(top5.result.df$value) * 0.88, 
           label = paste0("Median: ", round(median(top5.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "blue") +
  ylab("Top5") + xlab("") + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
ggsave(paste0(results.dir, "SynDef_result-top5.pdf"), width=7, height=2.5)

p6 <- ggplot(top10.result.df, aes(x=technique, y=value)) + geom_boxplot(fill="lightsteelblue") +
  stat_summary(fun = mean, geom = "point", shape = 17, size = 1.5, color = "red") + 
  annotate("text", x = 1, y = max(top10.result.df$value) * 0.95, 
           label = paste0("Mean: ", round(mean(top10.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "red") +
  annotate("text", x = 1, y = max(top10.result.df$value) * 0.88, 
           label = paste0("Median: ", round(median(top10.result.df$value), 3)), 
           hjust = 0.5, size = 3, color = "blue") +
  ylab("Top10") + xlab("") + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
ggsave(paste0(results.dir, "SynDef_result-top10.pdf"), width=7, height=2.5)

# Create combined metrics plot
combined_plot <- grid.arrange(p1, p2, p3, p4, p5, p6, ncol=3, nrow=2)
ggsave(paste0(results.dir, "SynDef_result-combined-metrics.pdf"), combined_plot, width=21, height=5)

print("SynDef_result analysis completed!")
print(paste0("Results saved to: ", results.dir))
print(paste0("Generated files:"))
print("- SynDef_results.csv")
print("- SynDef_statistics.csv")
print("- SynDef_result-Recall@Top20LOC.pdf")
print("- SynDef_result-Effort@Top20Recall.pdf")
print("- SynDef_result-IFA.pdf")
print("- SynDef_result-FPA.pdf")
print("- SynDef_result-top5.pdf")
print("- SynDef_result-top10.pdf")
print("- SynDef_result-combined-metrics.pdf")
print(paste0("Analysis timestamp: ", timestamp))
