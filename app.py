# import pickle
# import numpy as np
# import sys
# import os
# from sklearn.ensemble import RandomForestClassifier
# from flask import Flask, request

# port = int(os.environ.get('PORT', 5000))
#
# with open('./model.pkl', 'rb') as model_pkl:
#    clf = pickle.load(model_pkl)

# app = Flask(__name__)

print("Predicted result for observation " + "xx" + " is: " + "HOME TEAM WINS!")
# Create an API endpoint
# @app.route('/predict')


# def predict_if_home_team_wins():
# Read all necessary request parameters
# sl = request.args.get(‘sl’)
# sw = request.args.get(‘sw’)
# pl = request.args.get(‘pl’)
# pw = request.args.get(‘pw’)
# Use the predict method of the model to
# get the prediction for unseen data
# new_record = np.array([[sl, sw, pl, pw]])
# predict_result = knn.predict(new_record)
# return the result back
#    return 'Predicted result for observation ' + 'xx' + ' is...

# if __name__ == '__main__':
#    app.run(debug=True,host='0.0.0.0',port=port)
