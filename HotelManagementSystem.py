import os
import platform
import mysql.connector
import datetime

#mydb=mysql.connector.connect(host="localhost",user="root",passwd="root",database="hotel",charset="utf8")
def mysqlconnection():
    global mySqlDb
    mySqlDb = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='root',
    port=3306,
    database='lodge',
   # auth_plugin='mysql_native_password'
    )
#print(mySqlDb)
    mycursor1=mySqlDb.cursor()
    return mycursor1


def registerCustomerDetails():
    try:

        mycursor=mysqlconnection()

        hotelCustList = []

        custno = int(input('Enter customer no='))
        hotelCustList.append(custno)

        name = input('Enter Customer Name:')
        hotelCustList.append(name)

        addr = input('Enter Customer address:')
        hotelCustList.append(addr)

        roomBookdate = input('Enter Customer date of RoomBooking:yyyy-mm-dd')
        hotelCustList.append(roomBookdate)


        cust = (hotelCustList)



        sql = 'insert into cdata(custno,custname,addr,bdate) values(%s,%s,%s,%s)'
        mycursor.execute(sql, cust)
        mySqlDb.commit()
        mycursor.close()

    except Exception as e:
        print(e)

    finally:
        mySqlDb.close()


def roomRentCalculatation():
    try:

        mycursor=mysqlconnection()
        roomCostList = []

        cno = int(input('Enter customer  no='))
        roomCostList.append(cno)

        print('We have the following Besant Lodge rooms for you:-')
        print('1. type First class Room Cost--->Rs 6000 PN\-')
        print('2. type Business class Room cost around--->Rs 4000 PN\-')
        print('3. type Economy class Room Cost around --->Rs 2000 PN\-')

        choice= int(input('Enter your choice:'))
        n = int(input('Enter How many rooms you want ? '))

        if choice == 1:
            print('you have opted First class AC room.')
            s = 6000 * n
            roomCostList.append(s)
        elif choice == 2:
            print('you have opted First Class NON/AC room.')
            s = 4000 * n
            roomCostList.append(s)
        elif choice== 3:
            print('you have opted ordinary room.')
            s = 2000 * n
            roomCostList.append(s)
        else:
            print('Please select a class type.')

        print('your room rent amount  is =', s, '\n')
        print('Extra person per Head in room is Rs.500')

        y = int(input('How many  extra person stay yourself ? '))
        z = y * 500
        roomCostList.append(z)

        roomRent = (roomCostList)

        print('Your Totalbill:', s + z, '\n')
        g_tot = s + z

        roomCostList.append(g_tot)
        print(roomCostList)

        sql = "insert into roomrent (custno,rent_tot,ext_rent_tot,g_tot) values (%s,%s,%s,%s)"
        mycursor.execute(sql, roomRent)
        mySqlDb.commit()
        mycursor.close()

    except Exception as e:
        print(e)

    finally:
        mySqlDb.close()

def dispCustBill():
    try:

        mycursor = mysqlconnection()
        custno = int(input("Enter the customer number whose bill to be viewed : "))
        sql = "Select cdata.custno, cdata.custname, cdata.addr, roomrent.rent_tot, roomrent.ext_rent_tot, g_tot from cdata INNER JOIN roomrent ON cdata.custno=roomrent.custno and roomrent.custno = %s"
        rl = (custno,)


        mycursor.execute(sql, rl)
        res = mycursor.fetchall()

        for x in res:
            print(x)
        mycursor.close()

    except Exception as e:
        print(e)

    finally:
        mySqlDb.close()

def dispAllCustomerDetails():
    try:

        mycursor = mysqlconnection()
        sql = "Select cdata.custno, cdata.custname, cdata.addr,roomrent.rent_tot,roomrent.ext_rent_tot, g_tot from cdata INNER JOIN roomrent ON cdata.custno=roomrent.custno"
        mycursor.execute(sql)
        res = mycursor.fetchall()
        print("The Customer details are as follows : ")

        for x in res:
            print(x)
        mycursor.close()
    except Exception as e:
        print(e)

    finally:
        mySqlDb.close()

def Menuset():
    print('Enter 1: To enter customer data.')
    print('Enter 2: For ticketamount.')
    print('Enter 3: Display customerwise Details.')
    print('Enter 4: Display All Details.')
    print('Enter 5: Exit')

    userinput = int(input('Enter your choice:'))
    if userinput == 1:
        registerCustomerDetails()
    elif userinput == 2:
        roomRentCalculatation()
    elif userinput == 3:
        dispCustBill()
    elif userinput == 4:
        dispAllCustomerDetails()
    elif userinput == 5:
        quit()
    else:
        print('Sorry! ,Invalid Option! >> Enter correct choice.')


Menuset()


def runagain():
        runagn = input('\nWant to run again? y/n:')
        while runagn == 'y':
            if platform.system()=='Windows':
                print(os.system('cls'))
            else:
                print(os.system('clear'))

            Menuset()
            runagn=input('\nWant to run again? y/n:')
runagain()
