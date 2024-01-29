setwd("~/Downloads/")

library("survival")
library("survminer")

df <- read.csv("/Users/raghavtandon/Downloads/cox_df_subtype_0.0_delta_yrs_3.csv")

#df$convert <- factor(df$convert)
#df$APOE4 <- factor(df$APOE4)
#df$PTGENDER <- factor(df$PTGENDER)
#df <- df[df$time != 0,]
res.cox <- coxph(Surv(time,convert) ~ stages_inferred + AGE + PTGENDER + PTEDUCAT + APOE4, data =  df)
res.cox

summary(res.cox)
p <- ggforest(res.cox, data = NULL, main = "Hazard ratio", cpositions = c(0.02,0.22,0.4), 
              fontsize=0.7, refLabel = "reference", noDigits=2)
p 

png("~/Downloads/forest.png", units = "in", res=400, width = 6, height = 8)
print(({p}))
dev.off()

Call:
  coxph(formula = Surv(time, convert) ~ stages_inferred + AGE + 
          PTGENDER + PTEDUCAT + APOE4, data = df)

n= 551, number of events= 155 



coef exp(coef)  se(coef)      z Pr(>|z|)    
stages_inferred  0.455873  1.577551  0.059708  7.635 2.26e-14 ***
  AGE             -0.009137  0.990905  0.012969 -0.705   0.4811    
PTGENDERMale     0.304623  1.356114  0.180565  1.687   0.0916 .  
PTEDUCAT        -0.045567  0.955455  0.027831 -1.637   0.1016    
APOE4            0.560542  1.751622  0.115750  4.843 1.28e-06 ***
  ---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

exp(coef) exp(-coef) lower .95 upper .95
stages_inferred    1.5776     0.6339    1.4033     1.773
AGE                0.9909     1.0092    0.9660     1.016
PTGENDERMale       1.3561     0.7374    0.9519     1.932
PTEDUCAT           0.9555     1.0466    0.9047     1.009
APOE4              1.7516     0.5709    1.3961     2.198

Concordance= 0.718  (se = 0.022 )
Likelihood ratio test= 97.9  on 5 df,   p=<2e-16
Wald test            = 100.8  on 5 df,   p=<2e-16
Score (logrank) test = 110.1  on 5 df,   p=<2e-16

# Chisquare test for cetegories

df <- read.csv("chisq.csv")
df <- df[,-1]
print(chisq.test(df))

df2 <- df[,c("mci", "ad")]
chisq.test(df2)$p.value
