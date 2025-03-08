select * from dataset_1;

select passanger,weather from dataset_1;

select * from dataset_1 LIMIT 8;

select DISTINCT passanger from dataset_1;

select * from dataset_1 WHERE destination = 'Home';

select * from dataset_1 ORDER BY coupon;

select passanger as 'Passenger' from dataset_1;


select * from dataset_1 GROUP BY occupation;


select weather,
avg(temperature) as 'Average_Temperature'
from dataset_1
GROUP BY weather;


select weather, COUNT(temperature) as 'Count of Temparatures in Season'
from dataset_1
GROUP BY weather;


SELECT weather,COUNT(DISTINCT temperature) as 'Unique Temperatures'
from dataset_1
GROUP BY weather;

SELECT weather, SUM(temperature) as 'Total_temperature'
FROM dataset_1
GROUP BY weather;

SELECT weather, MIN(temperature) as 'Minimum_temperature'
FROM dataset_1
GROUP BY weather;

SELECT weather, MAX(temperature) as 'Maximum_temperature'
FROM dataset_1
GROUP BY weather;

SELECT occupation, AVG(age) as 'Average_Age'
FROM dataset_1
GROUP BY occupation
HAVING Average_Age > 36;

select * from table_to_union;

select * from dataset_1
UNION
select * from table_to_union;

select DISTINCT destination FROM
(
select * from dataset_1
UNION
select * from table_to_union
);

SELECT * from table_to_join;

select * FROM dataset_1 d
LEFT JOIN table_to_join ttj
on d.time = ttj.time ;

select *
FROM dataset_1 d
LEFT JOIN table_to_join ttj
on d.time = ttj.time ;

select dataset.destination as Destination, dataset.time as DataSet_time, joinTable.part_of_day as PartOfDay
FROM dataset_1 dataset
LEFT JOIN table_to_join joinTable
on dataset.time = joinTable.time ;


select * from dataset_1
WHERE passanger = 'Alone';

SELECT destination as Destination, passanger as Passenger
FROM
(SELECT * from dataset_1
WHERE passanger = 'Alone');

SELECT *
FROM dataset_1
WHERE weather LIKE 'Sun%';


SELECT temperature
FROM dataset_1
WHERE temperature BETWEEN 30 AND 75;


SELECT DISTINCT temperature
FROM dataset_1
WHERE temperature BETWEEN 30 AND 75;

SELECT occupation
FROM dataset_1 d 
WHERE occupation  IN ('Sales & Related','Management')


SELECT  destination,weather,
AVG(temperature) AS Average_Temperature
from dataset_1
GROUP BY temperature;

SELECT  destination,weather,
AVG(temperature) OVER (PARTITION BY weather) AS Average_Temperature
from dataset_1;


SELECT destination,weather
FROM dataset_1 d 
ORDER BY destination;

SELECT destination,weather,
ROW_NUMBER() OVER (PARTITION BY weather) as Row_Number
FROM dataset_1 d 
ORDER BY destination;

SELECT destination,weather,
ROW_NUMBER() OVER (PARTITION BY weather ORDER BY destination) as Row_Number
FROM dataset_1 d;


SELECT destination,weather,
RANK() OVER (PARTITION BY weather ORDER BY destination) as Row_Number
FROM dataset_1 d;


SELECT destination,weather,
DENSE_RANK() OVER (PARTITION BY weather ORDER BY destination) as Row_Number
FROM dataset_1 d;
