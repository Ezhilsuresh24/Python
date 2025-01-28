create database lodge;
create table cdata (custno varchar(50), custname varchar(15), addr varchar(50), bdate date);
create table roomrent (custno varchar(50), rent_tot decimal(10,2), ext_rent_tot decimal(10,2), g_tot decimal(10,2));
select * from cdata;
select * from roomrent;