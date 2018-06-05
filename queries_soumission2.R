# Load the package
library(RODBC)

### Two queries to create dummy variables for each month since 2007 with value of 1 if donation
### was made (separate for PA and DO)

# Connect to MySQL (use your own credentials)
db = odbcConnect("mysqlodbc", uid="root", pwd="root")
sqlQuery(db, "USE ma_charity_full")

# stats since 1 year

query = "select assignment2.*, month, year
from assignment2 
left join (select contact_id, month(acts.act_date) as month, year(acts.act_date) as year 
from acts where (acts.act_date>20070531 and acts.act_type_id='PA' and acts.campaign_id is not null) group by acts.contact_id, month, year) as x
on assignment2.contact_id=x.contact_id
group by assignment2.contact_id, month, year;"

data_pa = sqlQuery(db, query)

data_pa$y_m <- paste(data_pa[,'year'], data_pa[,'month'], sep="_") 
data_pa = within(data_pa, rm('year', 'month'))

for(level in unique(data_pa$y_m)){
  data_pa[paste("month", level, sep = "")] <- ifelse(data_pa$y_m == level, 1, 0)
}

write.csv(data_pa, file = "/Users/Jean/Desktop/Marketing Analytics/devoir2/final_try_pa.csv")

# stats since 1 year

query = "select assignment2.*, month, year
from assignment2 
left join (select contact_id, month(acts.act_date) as month, year(acts.act_date) as year 
from acts where (acts.act_date>20070531 and acts.act_type_id='DO' and acts.campaign_id is not null) group by acts.contact_id, month, year) as x
on assignment2.contact_id=x.contact_id
group by assignment2.contact_id, month, year;"

data_do = sqlQuery(db, query)

data_do$y_m <- paste(data_do[,'year'], data_do[,'month'], sep="_") 
data_do = within(data_do, rm('year', 'month'))

for(level in unique(data_do$y_m)){
  data_do[paste("month", level, sep = "")] <- ifelse(data_do$y_m == level, 1, 0)
}

write.csv(data_do, file = "/Users/Jean/Desktop/Marketing Analytics/devoir2/final_try_do.csv")

# Close the connection
odbcClose(db)



