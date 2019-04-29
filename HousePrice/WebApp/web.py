from flask import Flask, render_template, request
from Source.predictor import Predictor

app = Flask (__name__)
model = Predictor()

@app.route('/')
def home():
    return render_template('HouseValue.html')


@app.route('/results', methods =['POST', 'GET'])
def result():
    if request.method == 'POST':
        result =request.form
        state = request.form.getlist('myState')
        county = request.form.getlist('myCounty')
        city = request.form.getlist('myCity')
        neighborhood = request.form.getlist('myNeighborhood')
        sqf = request.form.getlist('sqFt')
        # print(state)
        # print(county)
        # print(city)
        # print(neighborhood)
        # print(result)
        val = model.predict(state=state[0], county=county[0], city=city[0], neighborhood=neighborhood[0], square_footage=int(sqf[0]))
        # print(val)
        return render_template("results.html", val = val)


if __name__ == '__main__':
    app.run(debug=False)