# Load the package
library(RODBC)

# Connect to MySQL (use your own credentials)
db = odbcConnect("mysqlodbc", uid="root", pwd="root")
sqlQuery(db, "USE ma_charity_full")

# import four queries, in order: DO variables in last 11 months, PA variables in last month,
# DO variables variables in June 2016 and response rate to campaigns

query = "select assignment2.*,
count(x.amount) as freq1,
avg(x.amount) as avg1,
sqrt(variance(x.amount)) as sd1
from assignment2 
left join (select * from acts where (acts.act_date>20160731 and acts.act_type_id='DO' and campaign_id is not null)) as x
on assignment2.contact_id=x.contact_id
group by assignment2.contact_id;"

recent2 = sqlQuery(db, query)

for(i in 6:8){
  recent2[is.na(recent2[,i]), i] <- 0
}

write.csv(recent2, file = "/Users/Jean/Desktop/Marketing Analytics/devoir2/recent2.csv")

###

query = "select assignment2.contact_id,
count(x.amount) as freq_pa,
avg(x.amount) as avg_pa,
sqrt(variance(x.amount)) as sd_pa
from assignment2 
left join (select * from acts where (acts.act_date>20170530 and acts.act_type_id='PA' and campaign_id is not null)) as x
on assignment2.contact_id=x.contact_id
group by assignment2.contact_id;"

pa2 = sqlQuery(db, query)

for(i in 1:ncol(pa2)){
  pa2[is.na(pa2[,i]), i] <- 0
}

write.csv(pa2, file = "/Users/Jean/Desktop/Marketing Analytics/devoir2/pa2.csv")

###

query = "select assignment2.contact_id,
count(x.contact_id) as freq_july16,
avg(x.amount) as avg_july16,
sqrt(variance(x.amount)) as sd_july16
from assignment2 
left join (select * from acts where (acts.act_type_id = 'DO' and month(acts.act_date)=07 and year(acts.act_date)=2016 and campaign_id is not null)) as x
on assignment2.contact_id=x.contact_id
group by assignment2.contact_id;"

july2 = sqlQuery(db, query)

for(i in 1:ncol(july2)){
  july2[is.na(july2[,i]), i] <- 0
}

write.csv(july2, file = "/Users/Jean/Desktop/Marketing Analytics/devoir2/july2.csv")

###

query = "select assignment2.contact_id, (count(x.campaign_id)/y.count) as resprate
from assignment2 
left join (select contact_id, campaign_id
           from acts  where (acts.act_date>20070531 and acts.campaign_id is not null) group by acts.contact_id, campaign_id) as x
on assignment2.contact_id=x.contact_id
left join (select contact_id, count(action_date) as count from actions where actions.action_date>20070431 group by contact_id) as y
on assignment2.contact_id=y.contact_id
group by assignment2.contact_id;"

resprate2 = sqlQuery(db, query)

for(i in 1:ncol(resprate2)){
  resprate2[is.na(resprate2[,i]), i] <- 0
}

write.csv(resprate2, file = "/Users/Jean/Desktop/Marketing Analytics/devoir2/resprate2.csv")

# Close the connection
odbcClose(db)

df <- merge(recent2, pa2, by="contact_id")
df <- merge(df, july2, by="contact_id")

# Assign contact id as row names, remove id from data
rownames(df) = df$contact_id
df = df[, -1]

# Create data set with just model features
df_model = df[!is.na(df[, 'donation']), c(2, 5:ncol(df))]
  
# One of the libraries available for (multinomial) logit model
library(nnet)

# Logit model
model = multinom(formula = donation ~ ., data = df_model)
AIC(model)
BIC(model)

# Get coefficients, standard errors
coeff = t(summary(model)$coefficients)
stder = t(summary(model)$standard.errors)
zvalues = coeff / stder
pvalues = (1 - pnorm(abs(zvalues), 0, 1)) * 2

# Print results
print("coefficients:")
print(coeff)
print("standard deviations:")
print(stder)
print("p-values")
print(pvalues)

# Out-of-sample predictions
df_pred = df[is.na(df[, 'donation']), c(5:ncol(df))]

out = data.frame(contactid = recent2[which(recent2$calibration==0),'contact_id'])
out$probs  = predict(object = model, newdata = df_pred, type = "probs")

# linear model

# In-sample, donation amount model
z = which(!is.na(data$targetamount))
print(head(data[z, ]))

df_pred = df[!is.na(df[, 'amount']), c(3, 5:ncol(df))]

model_continuous = lm(formula = amount ~ ., data = df_pred)
summary(model)

df_pred = df[is.na(df[, 'donation']), c(5:ncol(df))]
out$amount = predict(object = model_continuous, newdata = df_pred)

out$score  = out$probs * out$amount
  
# Show results
print(head(out))

# Who is likely to be worth more than 2 EUR?
# Select cut-off of 9 to get more realistic number of donators
z = which(out$score > 9)
out$reach = ifelse(out$score>9, 1, 0)

print(length(z))
print(length(z)/nrow(df[which(df[, 'calibration']==0),]))

result = data.frame(out[, c("contactid", "reach")])

write.table(data.frame(result), file = "/Users/Jean/Desktop/result2.txt", 
            sep = "\t", col.names = FALSE, row.names = FALSE)


### Metrics

# calculate MSE for each model

pred = predict(object = model, newdata = df[df[,'calibration']==1, c(5:ncol(df))], type = "probs")
mean((df[which(df[,'calibration']==1), 'donation']-pred)^2)
# 8% mse

pred_c = predict(object = model_continuous, newdata = df[!is.na(df[, 'amount']), c(5:ncol(df))])
mean((df[which(df[,'donation']==1), 'amount']-pred_c)^2)
head(pred_c)
head(df[which(df[,'donation']==1), 'amount'])
# 1975 mse

