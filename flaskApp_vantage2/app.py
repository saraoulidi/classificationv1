from pred import prediction
from flask import Flask, request
from flask_restplus import Api, Resource, fields

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Classification API", 
		  description = "Predict image classes")

name_space = app.namespace('main', description='Prediction APIs')
model = app.model('Post Prediction',
                  {'url': fields.String(required=True,
                                        description="Url to predict",
                                        help="Url cannot be blank.")})


@name_space.route("/")
class MainClass(Resource):
    @app.expect(model)
    def post(self):
        myPrediction = prediction(request.json['url'])
        return myPrediction


if __name__ == "__main__":
    app.run(debug=True)
