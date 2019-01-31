import mysql.connector
import numpy as np
import cv2
conn = mysql.connector.connect(user="root", password="", host="localhost")

cursor = conn.cursor(buffered=True)

# cursor.execute("CREATE DATABASE `rapidml`")
cursor.execute("USE `rapidml`")
# cursor.execute("CREATE TABLE `images`"
#                "("
#                 "IID int not null AUTO_INCREMENT PRIMARY KEY,"
#                 "Path varchar(255)"
#                ")")
# cursor.execute("CREATE TABLE `imglabel`"
#                "("
#                "IID int not null,"
#                "LID int not null"
#                ")")
# cursor.execute("CREATE TABLE `labels`"
#                "("
#                "LID int not null AUTO_INCREMENT PRIMARY KEY,"
#                "Label varchar(255)"
#                ")")

def add_image_name_with_extension(path):
    sql = "SELECT * FROM `images` WHERE Path = '{0}'".format(path)
    cursor.execute(sql)
    result = cursor.fetchall()
    if(len(result) == 0):
        sql = "INSERT INTO `images` (`Path`) VALUES ('{0}')".format(path)
        cursor.execute(sql)
        conn.commit()
    else:
        print("Table already updated")

def add_label_to_image(label, img):
    sql = "SELECT * FROM `labels` WHERE Label = '{0}'".format(label)
    cursor.execute(sql)
    result = cursor.fetchall()
    if(len(result) > 0):
        sql = "SELECT * FROM `images` WHERE Path = '{0}'".format(img)
        cursor.execute(sql)
        result1 = cursor.fetchall()
        sql = "SELECT * FROM `imglabel` WHERE IID = '{0}' AND LID = '{1}'".format(result1[0][0], result[0][0])
        cursor.execute(sql)
        result2 = cursor.fetchall()
        if(len(result2) == 0):
            sql = "INSERT INTO `imglabel`(`IID`, `LID`) VALUES ('{0}','{1}')".format(result1[0][0], result[0][0])
            cursor.execute(sql)
            conn.commit()
            print("All tables were updated")
        else:
            print("All tables are up to date")
    else:
        sql = "INSERT INTO `labels` (`Label`) VALUES ('{0}')".format(label)
        cursor.execute(sql)
        conn.commit()
        sql = "SELECT * FROM `labels` WHERE Label = '{0}'".format(label)
        cursor.execute(sql)
        result = cursor.fetchall()
        sql = "SELECT * FROM `images` WHERE Path = '{0}'".format(img)
        cursor.execute(sql)
        result1 = cursor.fetchall()
        sql = "INSERT INTO `imglabel`(`IID`, `LID`) VALUES ('{0}','{1}')".format(result1[0][0], result[0][0])
        cursor.execute(sql)
        conn.commit()
        print("All tables were updated")

def get_data_to_train():
    sql = "SELECT * FROM `labels`"
    cursor.execute(sql)
    result = cursor.fetchall()
    strlabel = []
    intlabel = []
    trainimg = []
    trainlabels = []
    trainstrlabels = []
    j = 0
    for res in (result):
        intlabel.append([])
        strlabel.append(res[1])
        for i in range(0,len(result)):
            if(i==(res[0]-1)):
                intlabel[j].append(1.0)
            else:
                intlabel[j].append(0.0)
        j+=1
    
    sql = "SELECT `labels`.`Label`, `images`.`Path` FROM `images`, `imglabel` INNER JOIN `labels` WHERE `labels`.`LID`= `imglabel`.LID LIMIT 20"
    cursor.execute(sql)
    result = cursor.fetchall()
    for res in result:
        path = "train/{0}".format(res[1])
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (128, 128), 0, 0, cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)

            # img = np.reshape(img, [1, -1])
            img = np.reshape(img, [1, 128, 128, -1])
            trainimg.append(img)

            trainlabels.append(intlabel[strlabel.index(res[0])])
            trainstrlabels.append(res[1])
        except: 
            print("Something went wrong", res[1])
    return {
            'images': trainimg, 
            'labels': trainlabels,
            'slabels': strlabel,
            'imgpath': trainstrlabels
            }

def get_connect():
    return [conn, cursor]

def get_train_info():
    labels = []
    images = []
    imglabel = []
    sql = "SELECT * FROM `labels`"
    cursor.execute(sql)
    result = cursor.fetchall()
    for res in result:
        labels.append(res[1])
    sql = "SELECT `images`.`IID`, `labels`.`Label` , `imglabel`.`LID`, `images`.`Path` FROM `images`, `imglabel` INNER JOIN labels WHERE `images`.IID = `imglabel`.IID and `imglabel`.`LID` = `labels`.`LID`"
    cursor.execute(sql)
    result = cursor.fetchall()
    for res in result:
        images.append(res[3])
        imglabel.append(res[1])
    return {
        "classes": labels,
        "images": images,
        "labels": imglabel
    }
        



get_train_info()



# cursor.execute("SELECT * FROM `test`")

# for (name, surname) in cursor:
#     print(name, surname)