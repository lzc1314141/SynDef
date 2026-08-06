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


save.fig.dir = 'D:/cursor code/SynDef/Hit_Over/'
result.dir = 'D:/cursor code/SynDef/Hit_Over/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(result.dir), showWarnings = FALSE, recursive = TRUE)

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

# Read DeepLineDP results for ground truth
prediction_dir = 'D:/cursor code/SPLICE-master/Baseline-result/DeepLineDP/output/prediction/DeepLineDP/within-release/'

all_files = list.files(prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

CEandNFCdir = "D:/cursor code/result_of_n-gram_PMD_and_LineDP/CEandNFCdir/test/"

all_CEandNF_files = list.files(CEandNFCdir)

df_CEandNF_all <- NULL

for(f in all_CEandNF_files)
{
  df <- read.csv(paste0(CEandNFCdir, f))
  df$test = str_split_fixed(f, "-result", 2)[,1]
  df_CEandNF_all  <- rbind(df_CEandNF_all, df)
}

df_CEandNF_all = select(df_CEandNF_all, "predicted_buggy_lines", "predicted_buggy_line_numbers","rank", "functioncall", "controlelements", "test")
names(df_CEandNF_all) = c("filename", "line.number", "rank", "NFC", "CE", "test")
df_CEandNF_all$filename = str_split_fixed(df_CEandNF_all$filename, ":", 2)[,1]

line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth, is.comment.line)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5 &  is.comment.line== "False")
line.ground.truth = distinct(line.ground.truth)

# Define evaluation releases
all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

# Read SynDef results
SynDef.result.dir = 'D:/cursor code/SynDef/SynDef_result/'

sorted_SynDef = NULL

for(rel in all_eval_releases)
{
  csv_file = paste0(SynDef.result.dir, rel, '-result.csv')
  
  if (!file.exists(csv_file)) {
    warning(paste("File not found:", csv_file))
    next
  }
  
  SynDef.result = read.csv(csv_file)
  
  # Extract filename from predicted_buggy_lines (format: filename:line_number)
  SynDef.result$filename = str_split_fixed(SynDef.result$predicted_buggy_lines, ":", 2)[,1]
  SynDef.result$line.number = as.numeric(SynDef.result$predicted_buggy_line_numbers)
  SynDef.result$test = rel
  
  # Merge with ground truth to get line.level.ground.truth
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
  
  SynDef.result = merge(SynDef.result, cur.df.file, by=c("filename", "line.number"), all.x = TRUE)
  
  # Sort by rank within each file and calculate order
  SynDef.result = SynDef.result %>% 
    group_by(test, filename) %>% 
    arrange(rank, .by_group = TRUE) %>% 
    mutate(order = row_number())
  
  sorted_SynDef = rbind(sorted_SynDef, SynDef.result)
  
  print(paste0('Loaded SynDef result for: ', rel))
}

# Process DeepLineDP results for comparison
print("Processing DeepLineDP results...")
print(paste0("Total rows in df_all: ", nrow(df_all)))

#Force attention score of comment line is 0
print("Setting comment line attention scores to 0...")
df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'
  
  return(top.k)
}

print("Extracting top-k tokens (this may take a while)...")
tmp.top.k = get.top.k.tokens(df_all, 1500)
print(paste0("Top-k tokens extracted: ", nrow(tmp.top.k), " rows"))

print("Merging with df_all (this may take a while)...")
merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)
print(paste0("Merge completed. Total rows: ", nrow(merged_df_all)))

merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0

## use top-k tokens 
print("Calculating line-level attention scores (this may take a while)...")
sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True" ) %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())
print(paste0("Line-level attention scores calculated: ", nrow(sum_line_attn), " rows"))

print("Merging CE/NFC metrics for SPLICE-F scoring...")
sum_line_attn = merge(sum_line_attn, df_CEandNF_all, by=c('test', 'filename', 'line.number'))
print("Sorting SPLICE-F results...")
sorted_SPLICE_F = sum_line_attn %>% filter(is.comment.line== "False") %>% group_by(test, filename) %>%  arrange(-attention_score * num_tokens * NFC, .by_group=TRUE) %>% mutate(order = row_number())
print("Sorting SPLICE-S results...")
sorted_SPLICE_S = sum_line_attn %>% filter(is.comment.line== "False") %>% group_by(test, filename) %>%  arrange(-CE, -NFC, -attention_score/num_tokens, .by_group=TRUE) %>% mutate(order = row_number())
print("Sorting SPLICE-G results...")
sorted_SPLICE_G = sum_line_attn %>% filter(is.comment.line== "False") %>% group_by(test, filename) %>% mutate(order = rank) %>% mutate(order = row_number())
print("Sorting DeepLineDP results...")
sorted_DeepLineDP = sum_line_attn %>% filter(is.comment.line== "False") %>% group_by(test, filename) %>%  arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())
print("DeepLineDP processing completed!")


########### Calculate the hit and over of TP, calculate the hit and over of TN; ############
########### report the results on each version, and report the overall results of all versions combined together #########
computeHitOver = function(baseSet, newSet){
  intersection = intersect(baseSet, newSet)
  diff = setdiff(newSet, baseSet)
  hit = nrow(intersection) / nrow(baseSet)
  over = nrow(diff) / nrow(baseSet)
  
  result = data.frame(hit = hit, over = over)
  return (result)
}

getHitOverResult = function(rel, baseList, newList){  
  TP_base = getTP(baseList) 
  TN_base = getTN(baseList)
  
  TP_new = getTP(newList) 
  TN_new = getTN(newList) 
  
  TP_result = computeHitOver(TP_base, TP_new)
  TN_result = computeHitOver(TN_base, TN_new)
  
  result = data.frame(test=rel, TP.hit = TP_result$hit, TP.over = TP_result$over, TN.hit = TN_result$hit, TN.over = TN_result$over)
  
  return(result)
}


getTP = function(lineList){  
  TP = lineList %>% group_by(test, filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2 & line.level.ground.truth=="True") %>% select(test, filename, line.number)
  
  return(TP)
}

getTN = function(lineList){  
  TN = lineList %>% group_by(test, filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort > 0.2 & line.level.ground.truth=="False") %>% select(test, filename, line.number)
  
  return(TN)
}

glance.lr.result.dir = 'D:/cursor code/SPLICE-master/Baseline-result/GLANCE/result/BASE-Glance-LR/line_result/test/'
n.gram.result.dir = 'D:/cursor code/result_of_n-gram_PMD_and_LineDP/n_gram_result/'
linedp.result.dir = 'D:/cursor code/result_of_n-gram_PMD_and_LineDP/MIT-LineDP-update/line_result/test/'
PMD.result.dir = 'D:/cursor code/result_of_n-gram_PMD_and_LineDP/PMD_result/'

# 新增方法目录
linedef.result.dir = 'D:/cursor code/LineDef/'
glance.ea.result.dir = 'D:/cursor code/result_of_n-gram_PMD_and_LineDP/GLANCE-EA_test/test/'
glance.md.result.dir = 'D:/cursor code/result_of_n-gram_PMD_and_LineDP/GLANCE-MD_test/test/'

glance.lr.hit.over = NULL
ngram.hit.over = NULL
deeplinedp.hit.over = NULL
linedp.hit.over = NULL
PMD.hit.over = NULL
spliceF.hit.over = NULL

# 新增方法hit.over变量
linedef.hit.over = NULL
glance.ea.hit.over = NULL
glance.md.hit.over = NULL
spliceS.hit.over = NULL
spliceG.hit.over = NULL
SynDef.hit.over = NULL

glance.lr.result.all = NULL
ngram.result.all = NULL
linedp.result.all = NULL
PMD.result.all = NULL

# 新增方法result.all变量
linedef.result.all = NULL
glance.ea.result.all = NULL
glance.md.result.all = NULL

print("Starting main processing loop...")
for(rel in all_eval_releases)
{  
  print(paste0("Processing release: ", rel))
  newList = sorted_SynDef[sorted_SynDef$test==rel, ]
  
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
  
  print(paste0("  Processing SPLICE-F for ", rel, "..."))
  spliceF.baseList = sorted_SPLICE_F[sorted_SPLICE_F$test==rel, ]
  temp = getHitOverResult(rel, spliceF.baseList, newList)
  spliceF.hit.over = rbind(spliceF.hit.over, temp)
  print(paste0("  SPLICE-F completed for ", rel))

  print(paste0("  Processing SPLICE-S for ", rel, "..."))
  spliceS.baseList = sorted_SPLICE_S[sorted_SPLICE_S$test==rel, ]
  temp = getHitOverResult(rel, spliceS.baseList, newList)
  spliceS.hit.over = rbind(spliceS.hit.over, temp)
  print(paste0("  SPLICE-S completed for ", rel))

  print(paste0("  Processing SPLICE-G for ", rel, "..."))
  spliceG.baseList = sorted_SPLICE_G[sorted_SPLICE_G$test==rel, ]
  temp = getHitOverResult(rel, spliceG.baseList, newList)
  spliceG.hit.over = rbind(spliceG.hit.over, temp)
  print(paste0("  SPLICE-G completed for ", rel))
  
  print(paste0("  Loading GLANCE-LR for ", rel, "..."))
  glance.lr.result = read.csv(paste0(glance.lr.result.dir,rel,'-result.csv'))
  glance.lr.result$filename = str_split_fixed(glance.lr.result$predicted_buggy_lines,":", 2)[,1]
  glance.lr.result = select(glance.lr.result,'filename',"predicted_buggy_line_numbers","rank")
  names(glance.lr.result) = c("filename", "line.number", "rank")
  glance.lr.result = merge(glance.lr.result, cur.df.file, by=c("filename", "line.number")) %>% mutate(test = rel) 
  glance.lr.result.all = rbind(glance.lr.result.all, glance.lr.result)
  
  baseList = glance.lr.result %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
  temp = getHitOverResult(rel, baseList, newList)
  glance.lr.hit.over  = rbind(glance.lr.hit.over, temp )
  print(paste0("  GLANCE-LR completed for ", rel))
  
  print(paste0("  Loading PMD for ", rel, "..."))
  PMD.result = read.csv(paste0(PMD.result.dir,rel,'-line-lvl-result.txt'),quote="")
  PMD.result$PMD_prediction_result <- ifelse(PMD.result$PMD_prediction_result == "False", 0, 1)
  PMD.result = PMD.result %>% group_by(filename) %>% arrange(-PMD_prediction_result, Priority, .by_group = TRUE) %>% mutate(rank = row_number())
  PMD.result = select(PMD.result,'filename','line_number','rank')
  names(PMD.result) = c('filename','line.number','rank')
  PMD.result = merge(PMD.result, cur.df.file, by=c("filename", "line.number")) %>% mutate(test = rel) 
  PMD.result.all = rbind(PMD.result.all, PMD.result)
  
  baseList = PMD.result %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
  temp = getHitOverResult(rel, baseList, newList)
  PMD.hit.over = rbind(PMD.hit.over, temp)
  print(paste0("  PMD completed for ", rel))
  
  print(paste0("  Loading N-gram for ", rel, "..."))
  ngram.result = read.csv(paste0(n.gram.result.dir,rel,'-line-lvl-result.txt'), sep = "\t", quote = "")
  ngram.result = select(ngram.result, "file.name", "line.number",  "line.score")
  ngram.result = distinct(ngram.result)
  ngram.result = ngram.result %>% group_by(file.name) %>% arrange(-line.score, .by_group = TRUE) %>% mutate(rank = row_number())
  ngram.result = select(ngram.result,'file.name','line.number','rank')
  names(ngram.result) = c('filename','line.number','rank')
  ngram.result = merge(ngram.result, cur.df.file, by=c("filename", "line.number")) %>% mutate(test = rel) 
  ngram.result.all = rbind(ngram.result.all, ngram.result)
  
  baseList = ngram.result %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
  temp = getHitOverResult(rel, baseList, newList)
  ngram.hit.over = rbind(ngram.hit.over, temp )
  print(paste0("  N-gram completed for ", rel))
  
  print(paste0("  Loading LineDP for ", rel, "..."))
  linedp.result = read.csv(paste0(linedp.result.dir,rel,'-result.csv'))
  linedp.result = select(linedp.result,'predicted_buggy_lines','rank')
  linedp.result$file.name = str_split_fixed(linedp.result$predicted_buggy_lines, ":", 2)[,1]
  linedp.result$line.numbers = str_split_fixed(linedp.result$predicted_buggy_lines, ":", 2)[,2]
  linedp.result$line.numbers = as.numeric(linedp.result$line.numbers)
  linedp.result = select(linedp.result,'file.name','line.numbers','rank')
  names(linedp.result) = c("filename", "line.number",'rank')
  linedp.result = merge(linedp.result, cur.df.file, by=c('filename','line.number')) %>% mutate(test = rel) 
  linedp.result.all = rbind(linedp.result.all, linedp.result)
  
  baseList = linedp.result %>% group_by(test,filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
  temp = getHitOverResult(rel,baseList, newList)
  linedp.hit.over = rbind(linedp.hit.over, temp)
  print(paste0("  LineDP completed for ", rel))
  
  # DeepLineDP hit and over
  print(paste0("  Processing DeepLineDP for ", rel, "..."))
  baseList = sorted_DeepLineDP[sorted_DeepLineDP$test==rel, ]
  temp = getHitOverResult(rel, baseList, newList)
  deeplinedp.hit.over = rbind(deeplinedp.hit.over, temp)
  print(paste0("  DeepLineDP completed for ", rel))

  # LineDef hit and over
  print(paste0("  Loading LineDef for ", rel, "..."))
  linedef.csv_file = paste0(linedef.result.dir, rel, "/", rel, "_result.csv")
  if (file.exists(linedef.csv_file)) {
    linedef.result = read.csv(linedef.csv_file)
    
    # Set token.attention.score to 0 for comment lines (不过滤，只设为0)
    linedef.result[linedef.result$is.comment.line == "True",]$token.attention.score = 0
    
    # Get top-k tokens
    tmp.top.k <- get.top.k.tokens(linedef.result, 1500)
    
    # Merge dataframes
    merged_linedef <- merge(linedef.result, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)
    merged_linedef[is.na(merged_linedef$flag),]$token.attention.score = 0
    
    # Summarize attention scores for lines (包含所有行包括注释行)
    sum_line_attn.linedef <- merged_linedef %>%
      group_by(filename, line.number) %>%
      summarize(attention_score = sum(token.attention.score), num_tokens = n(), .groups = "drop")
    
    # Sort by attention_score descending and assign ranks
    linedef.result = sum_line_attn.linedef %>%
      arrange(desc(attention_score)) %>%
      mutate(rank = row_number()) %>%
      select(filename, line.number, rank)
    
    linedef.result = merge(linedef.result, cur.df.file, by=c("filename", "line.number")) %>% mutate(test = rel)
    linedef.result.all = rbind(linedef.result.all, linedef.result)

    baseList = linedef.result %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
    temp = getHitOverResult(rel, baseList, newList)
    linedef.hit.over = rbind(linedef.hit.over, temp)
    print(paste0("  LineDef completed for ", rel))
  } else {
    warning(paste("LineDef file not found:", linedef.csv_file))
  }

  # GLANCE-EA hit and over
  print(paste0("  Loading GLANCE-EA for ", rel, "..."))
  glance.ea.result = read.csv(paste0(glance.ea.result.dir, rel, '-result.csv'))
  glance.ea.result$filename = str_split_fixed(glance.ea.result$predicted_buggy_lines, ":", 2)[,1]
  glance.ea.result = select(glance.ea.result, "filename", "predicted_buggy_line_numbers", "rank")
  names(glance.ea.result) = c("filename", "line.number", "rank")
  glance.ea.result = merge(glance.ea.result, cur.df.file, by=c("filename", "line.number")) %>% mutate(test = rel)
  glance.ea.result.all = rbind(glance.ea.result.all, glance.ea.result)

  baseList = glance.ea.result %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
  temp = getHitOverResult(rel, baseList, newList)
  glance.ea.hit.over = rbind(glance.ea.hit.over, temp)
  print(paste0("  GLANCE-EA completed for ", rel))

  # GLANCE-MD hit and over
  print(paste0("  Loading GLANCE-MD for ", rel, "..."))
  glance.md.result = read.csv(paste0(glance.md.result.dir, rel, '-result.csv'))
  glance.md.result$filename = str_split_fixed(glance.md.result$predicted_buggy_lines, ":", 2)[,1]
  glance.md.result = select(glance.md.result, "filename", "predicted_buggy_line_numbers", "rank")
  names(glance.md.result) = c("filename", "line.number", "rank")
  glance.md.result = merge(glance.md.result, cur.df.file, by=c("filename", "line.number")) %>% mutate(test = rel)
  glance.md.result.all = rbind(glance.md.result.all, glance.md.result)

  baseList = glance.md.result %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
  temp = getHitOverResult(rel, baseList, newList)
  glance.md.hit.over = rbind(glance.md.hit.over, temp)
  print(paste0("  GLANCE-MD completed for ", rel))
  
  print(paste0('✅ Finished processing release: ', rel))
}

print("Calculating overall hit/over for all projects...")
####GLANCE-LR hit and over#### 
print("  GLANCE-LR overall...")
glance.lr.baseList = glance.lr.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
temp = getHitOverResult("allprojects", glance.lr.baseList, sorted_SynDef)
glance.lr.hit.over = rbind(glance.lr.hit.over, temp)

####SPLICE-F hit and over####
print("  SPLICE-F overall...")
temp = getHitOverResult("allprojects", sorted_SPLICE_F, sorted_SynDef)
spliceF.hit.over = rbind(spliceF.hit.over, temp)

####SPLICE-S hit and over####
print("  SPLICE-S overall...")
temp = getHitOverResult("allprojects", sorted_SPLICE_S, sorted_SynDef)
spliceS.hit.over = rbind(spliceS.hit.over, temp)

####SPLICE-G hit and over####
print("  SPLICE-G overall...")
temp = getHitOverResult("allprojects", sorted_SPLICE_G, sorted_SynDef)
spliceG.hit.over = rbind(spliceG.hit.over, temp)

####LineDef hit and over####
print("  LineDef overall...")
linedef.baseList = linedef.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
temp = getHitOverResult("allprojects", linedef.baseList, sorted_SynDef)
linedef.hit.over = rbind(linedef.hit.over, temp)

####GLANCE-EA hit and over####
print("  GLANCE-EA overall...")
glance.ea.baseList = glance.ea.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
temp = getHitOverResult("allprojects", glance.ea.baseList, sorted_SynDef)
glance.ea.hit.over = rbind(glance.ea.hit.over, temp)

####GLANCE-MD hit and over####
print("  GLANCE-MD overall...")
glance.md.baseList = glance.md.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number())
temp = getHitOverResult("allprojects", glance.md.baseList, sorted_SynDef)
glance.md.hit.over = rbind(glance.md.hit.over, temp)

####PMD hit and over#####
print("  PMD overall...")
PMD.baseList = PMD.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
temp = getHitOverResult("allprojects", PMD.baseList, sorted_SynDef)
PMD.hit.over = rbind(PMD.hit.over, temp)


####ngram hit and over####
print("  N-gram overall...")
ngram.baseList = ngram.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
temp = getHitOverResult("allprojects", ngram.baseList, sorted_SynDef)
ngram.hit.over = rbind(ngram.hit.over, temp)


####linedp hit and over####
print("  LineDP overall...")
linedp.baseList = linedp.result.all %>% group_by(test, filename) %>% arrange(rank, .by_group = TRUE) %>% mutate(order = row_number()) 
temp = getHitOverResult("allprojects", linedp.baseList, sorted_SynDef)
linedp.hit.over = rbind(linedp.hit.over, temp)


####DeepLineDP hit and over####
print("  DeepLineDP overall...")
deepLineDP.baseList = sorted_DeepLineDP
temp = getHitOverResult("allprojects", deepLineDP.baseList , sorted_SynDef)
deeplinedp.hit.over = rbind(deeplinedp.hit.over, temp)

####SynDef hit and over####
print("  SynDef overall...")
SynDef.baseList = sorted_SynDef
temp = getHitOverResult("allprojects", SynDef.baseList, SynDef.baseList)
SynDef.hit.over = rbind(SynDef.hit.over, temp)
print("Overall calculations completed!")


print("Writing results to files...")
result.dir = 'D:/cursor code/SynDef/Hit_Over/'
dir.create(file.path(result.dir), recursive = TRUE, showWarnings = FALSE)

write.csv(glance.lr.hit.over, file = paste0(result.dir, 'GLANCE-LR.csv'), row.names = FALSE)
write.csv(spliceF.hit.over, file = paste0(result.dir, 'SPLICE-F.csv'), row.names = FALSE)
write.csv(spliceS.hit.over, file = paste0(result.dir, 'SPLICE-S.csv'), row.names = FALSE)
write.csv(spliceG.hit.over, file = paste0(result.dir, 'SPLICE-G.csv'), row.names = FALSE)
write.csv(linedef.hit.over, file = paste0(result.dir, 'LineDef.csv'), row.names = FALSE)
write.csv(glance.ea.hit.over, file = paste0(result.dir, 'GLANCE-EA.csv'), row.names = FALSE)
write.csv(glance.md.hit.over, file = paste0(result.dir, 'GLANCE-MD.csv'), row.names = FALSE)
write.csv(PMD.hit.over, file = paste0(result.dir, 'PMD.csv'), row.names = FALSE)
write.csv(ngram.hit.over, file = paste0(result.dir, 'N-gram.csv'), row.names = FALSE)
write.csv(deeplinedp.hit.over, file = paste0(result.dir, 'DeepLineDP.csv'), row.names = FALSE)
write.csv(linedp.hit.over, file = paste0(result.dir, 'LineDP.csv'), row.names = FALSE)
write.csv(SynDef.hit.over, file = paste0(result.dir, 'SynDef.csv'), row.names = FALSE)
print("CSV files written!")

########### prepare data for veen-figure plot  ###########
print("Preparing Venn diagram data...")

print("  GLANCE-LR TP/TN...")
TP = getTP(glance.lr.baseList)
TN = getTN(glance.lr.baseList)
write.table(TP, paste0(save.fig.dir,"glancelr-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"glancelr-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  SPLICE-F TP/TN...")
TP = getTP(sorted_SPLICE_F)
TN = getTN(sorted_SPLICE_F)
write.table(TP, paste0(save.fig.dir,"splice-F-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"splice-F-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  SPLICE-S TP/TN...")
TP = getTP(sorted_SPLICE_S)
TN = getTN(sorted_SPLICE_S)
write.table(TP, paste0(save.fig.dir,"splice-S-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"splice-S-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  SPLICE-G TP/TN...")
TP = getTP(sorted_SPLICE_G)
TN = getTN(sorted_SPLICE_G)
write.table(TP, paste0(save.fig.dir,"splice-G-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"splice-G-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  LineDef TP/TN...")
TP = getTP(linedef.baseList)
TN = getTN(linedef.baseList)
write.table(TP, paste0(save.fig.dir,"linedef-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"linedef-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  GLANCE-EA TP/TN...")
TP = getTP(glance.ea.baseList)
TN = getTN(glance.ea.baseList)
write.table(TP, paste0(save.fig.dir,"glance-ea-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"glance-ea-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  GLANCE-MD TP/TN...")
TP = getTP(glance.md.baseList)
TN = getTN(glance.md.baseList)
write.table(TP, paste0(save.fig.dir,"glance-md-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"glance-md-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  SynDef TP/TN...")
TP = getTP(sorted_SynDef)
TN = getTN(sorted_SynDef)
write.table(TP, paste0(save.fig.dir,"SynDef-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"SynDef-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  DeepLineDP TP/TN...")
TP = getTP(sorted_DeepLineDP)
TN = getTN(sorted_DeepLineDP)
write.table(TP, paste0(save.fig.dir,"deeplinedp-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"deeplinedp-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  PMD TP/TN...")
TP = getTP(PMD.baseList)
TN = getTN(PMD.baseList)
write.table(TP, paste0(save.fig.dir,"PMD-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"PMD-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  N-gram TP/TN...")
TP = getTP(ngram.baseList)
TN = getTN(ngram.baseList)
write.table(TP, paste0(save.fig.dir,"ngram-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"ngram-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("  LineDP TP/TN...")
TP = getTP(linedp.baseList)
TN = getTN(linedp.baseList)
write.table(TP, paste0(save.fig.dir,"linedp-TP.csv"), sep=",", col.names = FALSE, row.names = FALSE)
write.table(TN, paste0(save.fig.dir,"linedp-TN.csv"), sep=",", col.names = FALSE, row.names = FALSE)

print("All processing completed! ✅")

