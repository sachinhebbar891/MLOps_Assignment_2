import os
import joblib
from flask import Blueprint, jsonify, request
from flask import current_app as app

bp = Blueprint("root", __name__)
base_dir = os.path.abspath(os.path.dirname(__file__))
model_file_rel_path = "model/titanic_model.pkl"
model_file_path = os.path.join(base_dir, model_file_rel_path)
model = joblib.load(model_file_path)


@bp.route("/predictions", methods=['POST'])
def index():
    print("Model path: %s" % model_file_path)
    input_json = request.get_json(force=True)
    # Pclass	Sex	   Age	SibSp	Parch      	Fare	Embarked_C       Embarked_Q        Embarked_S
    pclass = input_json.get('pclass')
    sex = input_json.get('sex')
    age = input_json.get('age')
    sibsp = input_json.get('sibsp')
    parch = input_json.get('parch')
    fare = input_json.get('fare')
    embarked_c = input_json.get('embarked_c')
    embarked_q = input_json.get('embarked_q')
    embarked_s = input_json.get('embarked_s')
    input_data = [pclass, sex, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s]
    prediction = model.predict([input_data])
    app.logger.debug("Input from user: %s" % prediction)
    if prediction[0] == 1:
        return jsonify({'prediction': 'Survived'})
    else:
        return jsonify({'prediction': 'Not Survived'})
