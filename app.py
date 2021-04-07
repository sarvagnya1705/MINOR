from flask import Flask, render_template, request
import jsonify
import requests
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('logistic_regression2.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


sc = StandardScaler()

@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
        #Year = int(request.form['Year'])
        #Present_Price=float(request.form['Present_Price'])
        #Kms_Driven=int(request.form['Kms_Driven'])
        #Kms_Driven2=np.log(Kms_Driven)
        #Owner=int(request.form['Owner'])
        #Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        #if(Fuel_Type_Petrol=='Petrol'):
                #Fuel_Type_Petrol=1
                #Fuel_Type_Diesel=0
        #else:
            #Fuel_Type_Petrol=0
            #Fuel_Type_Diesel=1
        #Year=2020-Year
        #Seller_Type_Individual=request.form['Seller_Type_Individual']
        #if(Seller_Type_Individual=='Individual'):
            #Seller_Type_Individual=1
        #else:
            #Seller_Type_Individual=0 */	
        #Transmission_Mannual=request.form['Transmission_Mannual']
        #if(Transmission_Mannual=='Mannual'):
            #Transmission_Mannual=1
        #else:
            #Transmission_Mannual=0
        
        X_test=[[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,	compactness_mean,	concavity_mean,	concave_points_mean	,symmetry_mean,	fractal_dimension_mean,	radius_se,	texture_se,	perimeter_se,	area_se,	smoothness_se,	compactness_se,	concavity_se,	concave_points_se,	symmetry_se,	fractal_dimension_se,	radius_worst,	texture_worst,	perimeter_worst,	area_worst,	smoothness_worst,	compactness_worst,	concavity_worst,	concave_points_worst,	symmetry_worst,	fractal_dimension_worst]
        ,[17.99,	10.38,	122.80,	1001.0,	0.11840,	0.27760,	0.30010,	0.14710,	0.2419,	0.07871,	1.0950,	0.9053,	8.589,	153.40,	0.006399,	0.04904,	0.05373,	0.015870,	0.03003,	0.006193	,25.38,	17.33,	184.60,	2019.0,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.11890]
        ,[20.57,	17.77,	132.9,	1326,	0.08474	,0.07864,	0.0869,	0.07017,	0.1812,	0.05667,	0.5435,	0.7339,	3.398,	74.08,	0.005225,	0.01308	,0.0186	,0.0134	,0.01389,	0.003532	,24.99	,23.41	,158.8	,1956	,0.1238,	0.1866,	0.2416	,0.186	,0.275	,0.08902]
        ,[11.42	,20.38,	77.58	,386.1,	0.1425,	0.2839,	0.2414,	0.1052,	0.2597,	0.09744,	0.4956,	1.156,	3.445,	27.23,	0.00911	,0.07458,	0.05661,	0.01867,	0.05963	,0.009208	,14.91	,26.5,	98.87,	567.7	,0.2098,	0.8663,	0.6869,	0.2575,	0.6638,	0.173]
        ,[13.54,	14.36,	87.46	,566.3,	0.09779,	0.08129,	0.06664,	0.04781,	0.1885,	0.05766,	0.2699,	0.7886,	2.058,	23.56,	0.008462,	0.0146,	0.02387	,0.01315	,0.0198,	0.0023,	15.11,	19.26,	99.7,	711.2,	0.144,	0.1773,	0.239,	0.1288,	0.2977,	0.07259]
        ,[16.65	,21.38,	110,	904.6	,0.1121	,0.1457,	0.1525	,0.0917,	0.1995,	0.0633	,0.8068,	0.9017,	5.455,	102.6,	0.006048	,0.01882,	0.02741,	0.0113,	0.01468,	0.002801	,26.46	,31.56	,177,	2215	,0.1805	,0.3578	,0.4695,	0.2095,	0.3613	,0.09564]
        ,[17.57	,15.05,	115	,955.1,	0.09847,	0.1157,	0.09875,	0.07953,	0.1739,	0.06149	,0.6003,	0.8225,	4.655,	61.1,	0.005627,	0.03033,	0.03407,	0.01354,	0.01925,	0.003742,	20.01,	19.52,	134.9,	1227,	0.1255,	0.2812,	0.2489,	0.1456,	0.2756,	0.07919]
        ,[13.03,	18.42,	82.61,	523.8,	0.08983,	0.03766,	0.02562,	0.02923,	0.1467	,0.05863	,0.1839	,2.342	,1.17	,14.16	,0.004352	,0.004899	,0.01343,	0.01164	,0.02671	,0.001777,	13.3	,22.81	,84.46	,545.9	,0.09701,	0.04619,	0.04833,	0.05013,	0.1987,	0.06169]
        ,[18.65,	17.6,	123.7,	1076,	0.1099,	0.1686,	0.1974,	0.1009,	0.1907,	0.06049,	0.6289,	0.6633,	4.293,	71.56,	0.006294,	0.03994,	0.05554,	0.01695,	0.02428,	0.003535,	22.82,	21.32	,150.6,	1567,	0.1679,	0.509,	0.7345,	0.2378,	0.3799,	0.09185]
        ,[11.94,	18.24,	75.71,	437.6,	0.08261,	0.04751,	0.01972,	0.01349,	0.1868	,0.0611,	0.2273,	0.6329,	1.52,	17.47,	0.00721,	0.00838,	0.01311,	0.008	,0.01996,	0.002635,	13.1,	21.33,	83.67,	527.2	,0.1144,	0.08906	,0.09203,	0.06296,	0.2785,	0.07408]
        ,[15.1,	22.02,	97.26,	712.8,	0.09056,	0.07081,	0.05253	,0.03334,	0.1616	,0.05684,	0.3105,	0.8339,	2.097,	29.91,	0.004675,	0.0103,	0.01603,	0.009222,	0.01095,	0.001629,	18.1,	31.69,	117.7	,1030	,0.1389,	0.2057,	0.2712,	0.153,	0.2675,	0.07873]
        ,[9.029,	17.33,	58.79,	250.5,	0.1066,	0.1413,	0.313,	0.04375,	0.2111,	0.08046,	0.3274,	1.194	,1.885,	17.67,	0.009549,	0.08606,	0.3038,	0.03322,	0.04197,	0.009559,	10.31,	22.65	,65.5,	324.7,	0.1482	,0.4365,	1.252	,0.175	,0.4228,	0.1175]
        ]
        print(X_test)
        X_test1 = sc.fit_transform(X_test)
        print(X_test1)
        prediction=model.predict(X_test1)
        output=prediction[0]
         
       
        if output == 0:
          return render_template('index.html',prediction_texts="Cancer Detected!!! contact your Doctor")
        else:
          return render_template('index.html',prediction_texts="congratulations!!! No cancer detected")
        
           
       
              
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
