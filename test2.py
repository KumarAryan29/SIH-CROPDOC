from flask import Flask, request, render_template
from meta_ai_api import MetaAI

app = Flask(__name__)

@app.route('/')
def home():
    # Get the query from the URL
    query = request.args.get('query')

    # Use the MetaAI API to get the response
    ai = MetaAI()
    response = ai.prompt(message=query)
    
    # Render the HTML page with the response
    return render_template('try.html', query=query, response=response)

if __name__ == '__main__':
    app.run(debug=True)
