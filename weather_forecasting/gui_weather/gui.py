from flask import Flask, render_template, flash, redirect, request, url_for, send_file, session
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm  import SVC
from sklearn.metrics import classification_report,confusion_matrix,r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

app = Flask(__name__)
app.secret_key = "dhutrr"


@app.route('/',methods = ['GET','POST'])
def hello_world():
    session.permanent = True 
    session["all"]=[]
    return render_template("weather.html")

@app.route('/weather',methods = ['POST','GET'])
def weather():
    if request.method=='POST':
        pass
        xattributes  = ['Average_humidity', 'Average_dewpoint',
       'Average_barometer', 'Average_windspeed', 'Average_gustspeed',
       'Average_direction', 'Rainfall_for_month', 'Rainfall_for_year',
       'Maximum_rain_per_minute', 'Maximum_temperature', 'Minimum_temperature',
       'Minimum_pressure', 'Maximum_pressure', 'Maximum_windspeed',
       'Maximum_gust_speed', 'Maximum_humidity', 'Minimum_humidity',
       'Maximum_heat_index']

        x_col  = []

        for att in xattributes:
            attr = request.form.get(att)
            if(attr!=None):
                x_col.append(attr)
        print("X_col : ",x_col)

    return render_template("weather1.html",xvalues = x_col)

@app.route('/weather2',methods = ['POST','GET'])
def weather2():
    if request.method=='POST':
        all_col  = []

        xattributes  = ['Average_humidity', 'Average_dewpoint',
       'Average_barometer', 'Average_windspeed', 'Average_gustspeed',
       'Average_direction', 'Rainfall_for_month', 'Rainfall_for_year',
       'Maximum_rain_per_minute', 'Maximum_temperature', 'Minimum_temperature',
       'Minimum_pressure', 'Maximum_pressure', 'Maximum_windspeed',
       'Maximum_gust_speed', 'Maximum_humidity', 'Minimum_humidity',
       'Maximum_heat_index']

        Model = []

        for att in xattributes:
            attr = request.form.get(att)
            if(attr!=None):
                all_col.append(attr)
        print("All_col : ",all_col)
        x_all_col = [ val.split("+")[1] for val in all_col if val.split("+")[0]=='x']
        y_all_col = [ val.split("+")[1] for val in all_col if val.split("+")[0]=='y']

        print("x_col_next : ",x_all_col)
        print("y_col_next : ",y_all_col)

        df3 = pd.read_csv("weather6.csv")
    
        
        x_train,x_test,y_train,y_test = train_test_split(df3[['Average_humidity','Average_dewpoint','Average_barometer']],df3[['Maximum_temperature']],test_size=0.2)

        # Linear Regression
        print("Linear regression ++++++++++++++++++++=+++++++++++++++++++++++")
        lr= LinearRegression()
        lr.fit(x_train,y_train)
        y_pred = lr.predict(x_test)
        print("Linear regression Loss : ",r2_score(y_test,y_pred))

        Model.append({
            "accuracy": str(100-100*float(r2_score(y_test,y_pred))),
            "loss":r2_score(y_test,y_pred),
            "model":"linear Regression"
        })



        df3['Maximum_temperature'].describe()['mean']
        for y_y in y_all_col:
            l2=[]
            for i,j in enumerate(df3[y_y]):
                l2.append(1 if j>df3[y_y].describe()['mean'] else 0)
            df3[y_y] = l2
       #SVM===========  
        if(len(y_all_col)==1):
            x_train,x_test,y_train,y_test=train_test_split(df3[x_all_col],df3[y_all_col],test_size=0.2,random_state=210)
            print("SVM ++++++++++++++++++++=+++++++++++++++++++++++")
            svcc = SVC(kernel='linear')
            svcc.fit(x_train,y_train)
            y_pred = svcc.predict(x_test)
            print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
            # print(confusion_matrix(y_test,y_pred))
            # print(classification_report(y_test,y_pred))
            print("Svm Loss : ",r2_score(y_test,y_pred))

            Model.append({
                "accuracy": metrics.accuracy_score(y_test, y_pred),
                "loss":r2_score(y_test,y_pred),
                "model":"SVM"
            })

        #Gaussian=====================
            print("Gaussian NB ++++++++++++++++++++=+++++++++++++++++++++++")
            x_train,x_test,y_train,y_test=train_test_split(df3[x_all_col],df3[y_all_col],test_size=0.2,random_state=50)
            model = GaussianNB()
            model.fit(x_train,y_train)
            ypred = model.predict(x_test)
            print("Gaussian Accuracy:",metrics.accuracy_score(y_test, ypred))
            # print(confusion_matrix(y_test,ypred))
            # print(classification_report(y_test,ypred))
            print("Gaussian Loss : ",r2_score(y_test,ypred))

            Model.append({
                "accuracy": metrics.accuracy_score(y_test, ypred),
                "loss":r2_score(y_test,ypred),
                "model":"Gaussian NB"
            })

        #Random forest==========
        print("Random forest ++++++++++++++++++++=+++++++++++++++++++++++")
        x_train,x_test,y_train,y_test=train_test_split(df3[x_all_col],df3[y_all_col],test_size=0.2,random_state=210)
        clf = RandomForestClassifier(max_depth=20, random_state=0)
        clf.fit(x_train, y_train)
        ypred1 = clf.predict(x_test)
        print("Random Accuracy:",metrics.accuracy_score(y_test, ypred1))
        # print(confusion_matrix(y_test,ypred1))
        # print(classification_report(y_test,ypred1))
        print("Random Loss :",r2_score(y_test,ypred1))

        Model.append({
            "accuracy": metrics.accuracy_score(y_test, ypred1),
            "loss":r2_score(y_test,ypred1),
            "model":"Random Forest"
        })

        #desicion tree===============

        print("Decision Tree ++++++++++++++++++++=+++++++++++++++++++++++")
        x_train,x_test,y_train,y_test=train_test_split(df3[x_all_col],df3[y_all_col],test_size=0.2,random_state=50)
        dtree = DecisionTreeClassifier()
        dtree.fit(x_train,y_train)
        pred1=dtree.predict(x_test)
        print("Decision Treee Accuracy :",metrics.accuracy_score(y_test,pred1))
        # print(confusion_matrix(y_test,y_pred))
        # print(classification_report(y_test,pred1))
        print("Decision Tree Loss :",r2_score(y_test,pred1))

        Model.append({
            "accuracy": metrics.accuracy_score(y_test, pred1),
            "loss":r2_score(y_test,pred1),
            "model":"Decision Tree"
        })

        
    return render_template("weather3.html",Model = Model)



if __name__ == "__main__":
    app.run(debug=True)


