
from flask import Flask, render_template,Response
import psycopg2
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)



# Home page route
@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/plot_heat', methods=['GET', 'POST'])
def plot_heat():
    
    conn = psycopg2.connect(
            host="localhost",
            port='5432',
            database="SQLtest",
            user='postgres',
            password='Fr57a7emd123*')

    # Open a cursor to perform database operations
    cur = conn.cursor()
    
    # Execute a command: this creates a new table
    cur.execute('select heat_type,count(heat_type) from interior_info group by heat_type order by heat_type asc')

    data = cur.fetchall()

    conn.commit()

        # Close the database connection
    cur.close()
    conn.close()

    #plotting barplot
    x = [pair[0] for pair in data]
    y = [pair[1] for pair in data]

    plt.bar(x, y)
    plt.title('Heat Type Graph')
    plt.xlabel('Heat Type')
    plt.ylabel('# Buildings')
    
    # Convert the plot to a PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    

    # Return the image as a Flask response with the appropriate content type
    return Response(img.getvalue(), mimetype='image/png')



###########################################################################################################

@app.route('/plot_basement', methods=['GET', 'POST'])
def plot_basement():
    
    conn = psycopg2.connect(
            host="localhost",
            port='5432',
            database="SQLtest",
            user='postgres',
            password='Fr57a7emd123*')

    # Open a cursor to perform database operations
    cur = conn.cursor()
    
    # Execute a command: this creates a new table
    cur.execute('select basement_type,count(basement_type) from interior_info group by basement_type order by basement_type asc')
    
    data = cur.fetchall()
   
    conn.commit()
    
    # Close the database connection
    cur.close()
    conn.close()

    #plotting barplot
    x = [pair[0] for pair in data]
    y = [pair[1] for pair in data]

    plt.bar(x, y)
    plt.title('Basement Type Graph')
    plt.xlabel('Basement Type')
    plt.ylabel('# Buildings')
    
    # Convert the plot to a PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Return the image as a Flask response with the appropriate content type
    return Response(img.getvalue(), mimetype='image/png')

# # ####################################################################################################################




if __name__ == '__main__':
    app.run(debug=True)


