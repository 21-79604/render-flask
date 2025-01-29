from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import base64
import io
app = Flask(__name__)

import os

# Get the current working directory (where the app.py file is located)
base_dir = os.path.abspath(os.path.dirname(__file__))

# Define model paths relative to the app's directory
binary_model_path = os.path.join(base_dir, 'complete_model.h5')
breed_model_path = os.path.join(base_dir, 'dogbreeds.keras')
binary_model = tf.keras.models.load_model(binary_model_path)
breed_model = tf.keras.models.load_model(breed_model_path)

# Class names for breed classification
class_names = [
    "Afghan_hound", "African_hunting_dog", "Airedale", "American_Staffordshire_terrier",
    "Appenzeller", "Askal", "Australian_terrier", "Bedlington_terrier", "Bernese_mountain_dog",
    "Blenheim_spaniel", "Border_collie", "Border_terrier", "Boston_bull", "Bouvier_des_Flandres",
    "Brabancon_griffon", "Brittany_spaniel", "Cardigan", "Chesapeake_Bay_retriever", "Chihuahua",
    "Dandie_Dinmont", "Doberman", "English_foxhound", "English_setter", "English_springer", "EntleBucher",
    "Eskimo_dog", "French_bulldog", "German_shepherd", "German_short-haired_pointer", "Gordon_setter",
    "Great_Dane", "Great_Pyrenees", "Greater_Swiss_Mountain_dog", "Ibizan_hound", "Irish_setter",
    "Irish_terrier", "Irish_water_spaniel", "Irish_wolfhound", "Italian_greyhound", "Japanese_spaniel",
    "Kerry_blue_terrier", "Labrador_retriever", "Lakeland_terrier", "Leonberg", "Lhasa", "Maltese_dog",
    "Mexican_hairless", "Newfoundland", "Norfolk_terrier", "Norwegian_elkhound", "Norwich_terrier",
    "Old_English_sheepdog", "Pekinese", "Pembroke", "Pomeranian", "Rhodesian_ridgeback", "Rottweiler",
    "Saint_Bernard", "Saluki", "Samoyed", "Scotch_terrier", "Scottish_deerhound", "Sealyham_terrier",
    "Shetland_sheepdog", "Shih-Tzu", "Siberian_husky", "Staffordshire_bullterrier", "Sussex_spaniel",
    "Tibetan_mastiff", "Tibetan_terrier", "Walker_hound", "Weimaraner", "Welsh_springer_spaniel",
    "West_Highland_white_terrier", "Yorkshire_terrier", "affenpinscher", "basenji", "basset", "beagle",
    "black-and-tan_coonhound", "bloodhound", "bluetick", "borzoi", "boxer", "briard", "bull_mastiff",
    "cairn", "chow", "clumber", "cocker_spaniel", "collie", "curly-coated_retriever", "dhole", "dingo",
    "flat-coated_retriever", "giant_schnauzer", "golden_retriever", "groenendael", "keeshond", "kelpie",
    "komondor", "kuvasz", "malamute", "malinois", "miniature_pinscher", "miniature_poodle",
    "miniature_schnauzer", "otterhound", "papillon", "pug", "redbone", "schipperke", "silky_terrier",
    "soft-coated_wheaten_terrier", "standard_poodle", "standard_schnauzer", "toy_poodle", "toy_terrier",
    "vizsla", "whippet", "wire-haired_fox_terrier"
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load and process the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')  # Convert to RGB

        # Resize image for binary classification
        binary_image_size = (150, 150)
        image_resized_for_binary = image.resize(binary_image_size)
        image_array = np.array(image_resized_for_binary) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction for binary classification
        binary_predictions = binary_model.predict(image_array)
        is_dog = binary_predictions[0][0] <= 0.5  # Adjust based on your model's output

        if is_dog:
            # Resize image for breed classification
            breed_image_size = (128, 128)
            image_resized_for_breed = image.resize(breed_image_size)
            breed_image_array = np.array(image_resized_for_breed) # Normalize the image
            breed_image_array = np.expand_dims(breed_image_array, axis=0)  # Add batch dimension
            
            # Make prediction for breed classification
            breed_predictions = breed_model.predict(breed_image_array)
            predicted_breed_index = np.argmax(breed_predictions, axis=1)[0]
            predicted_breed_name = class_names[predicted_breed_index]

            return jsonify({'predicted_class': 'Dog', 'breed': predicted_breed_name})
        else:
            return jsonify({'predicted_class': 'Not a Dog'})
    
    except Exception as e:
        return jsonify({'error': str(e)})
# Load ResNet50 model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Create a connection to the MySQL database
mydb = mysql.connector.connect(
    host="srv1757.hstgr.io",
    user="u287105369_admin",
    password='Batak_1234',
    database="u287105369_strayfree"  # Name of the database
)

# Create a cursor object
cursor = mydb.cursor()
cursor1 = mydb.cursor()
cursor2 = mydb.cursor()

# Prepare the query to retrieve images, colors, names, and breeds
query = 'SELECT * FROM captured_dogs'
cursor.execute(query)
data = cursor.fetchall()

queryDog = """
    SELECT * 
    FROM missing_dog_reports AS mdr
    JOIN dogs AS d ON mdr.dog_id = d.id
"""
cursor1.execute(queryDog)
data1 = cursor1.fetchall()


Dogincident = 'SELECT * FROM incidents'
cursor2.execute(Dogincident)
data2 = cursor2.fetchall()

def extract_features(img, model):
    img = img.resize((224, 224))  # Resize image
    img_data = np.array(img)  # Convert to numpy array
    img_data = np.expand_dims(img_data, axis=0)
    img_data = tf.keras.applications.resnet50.preprocess_input(img_data)
    features = model.predict(img_data)
    return features

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_content = file.read()
        uploaded_image = Image.open(io.BytesIO(file_content))
        uploaded_image.verify()  # Verify the image format
        uploaded_image = Image.open(io.BytesIO(file_content))  # Reopen after verification

        # Set similarity threshold
        similarity_threshold = 0.1 

        # Retrieve and process image BLOBs from the database
        database_features = []
        database_image_blobs = []
        database_colors = []
        database_names = []
        database_prediction = []
        database_dateOfCaptured = []
        database_captured_at = []
        database_status=[]

        for row in data:
            image_blob = row[5]
            color = row[3]
            name = row[1]
            prediction = row[2]
            dateOfCaptured= row[4]
            captured_at = row[7]
            status = row [8]
            try:
                encoded_image_blob = base64.b64encode(image_blob).decode('utf-8')
                binary_data = base64.b64decode(encoded_image_blob)
                
                img_image = Image.open(io.BytesIO(binary_data))

                # Extract features for each image
                features = extract_features(img_image, base_model)
                if features is not None and features.size > 0:
                    database_features.append(features)
                    database_image_blobs.append(binary_data)
                    database_colors.append(color)
                    database_names.append(name)
                    database_prediction.append(prediction)
                    database_dateOfCaptured.append(dateOfCaptured)
                    database_captured_at.append(captured_at)
                    database_status.append(status)

                else:
                    print(f"Empty features for database image")
            except Exception as e:
                print(f"Error processing database image: {e}")

        if not database_features:
            return jsonify({'error': 'No valid images found in the database'}), 500

        # Extract features from the uploaded image
        uploaded_features = extract_features(uploaded_image, base_model)

        # Calculate similarities
        similarities = [cosine_similarity(uploaded_features, db_feat).flatten()[0] for db_feat in database_features]

        if not similarities:
            return jsonify({'error': 'No similarities calculated'}), 500

        # Filter out scores below the similarity threshold
        filtered_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]

        if not filtered_indices:
            return jsonify({'error': 'No similar images found with score above threshold'}), 404

        # Find the indices of the top 5 most similar images that meet the threshold
        top_n_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)[:5]

        similar_images = []
        for index in top_n_indices:
            similar_images.append({
                'image': base64.b64encode(database_image_blobs[index]).decode('utf-8'),
                'similarity_score': float(similarities[index]),
                'color': database_colors[index],
                'dogname': database_names[index],
                'breed': database_prediction[index],
                'dateOfCaptured' : database_dateOfCaptured [index],
                'captured_at' :  database_captured_at [index],
                'status' :  database_status [index]
            })

        return jsonify({'similar_images': similar_images})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_images():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON format'}), 400

    name_query = data.get('name', '')

    if not name_query:
        return jsonify({'error': 'Name parameter is required'}), 400

    # Log the received name query
    print(f"Received search query: {name_query}")

    # Prepare the query to retrieve images, colors, names, and breeds
    query = 'SELECT * FROM captured_dogs WHERE dogname LIKE %s'
    params = [f'%{name_query}%']
    
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()

    except Exception as e:
        return jsonify({'error': f'Database query error: {e}'}), 500

    if not results:
        return jsonify({'message': 'No images found matching the name'}), 404


    similar_images = []
    for idx, row in enumerate(results):
        image = row[5]
        color = row[3]
        dogname = row[1]
        breed = row[2]
        dateOfCaptured= row[4]
        captured_at = row[7]
        status = row [8]
        qrcode = row[6]

   
        # Save the image for debugging
        encoded_image = base64.b64encode(image).decode('utf-8')


        if qrcode:
            try:
                encoded_Qrcode = base64.b64encode(qrcode).decode('utf-8')
            except Exception as e:
                print(f"Error encoding QR code: {e}")
                encoded_Qrcode = None
        else:
            encoded_Qrcode = None
        # Encode each image_blob individually
       
        similar_images.append({
            'image': encoded_image,
            'color': color,
            'dogname': dogname,
            'breed': breed,
            'dateOfCaptured': dateOfCaptured ,
            'captured_at': captured_at,
            'status': status,
            'qrcode': encoded_Qrcode,
        })

    return jsonify({'similar_images': similar_images})

@app.route('/findMyDog', methods=['POST'])
def upload_images():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_content = file.read()
        uploaded_image = Image.open(io.BytesIO(file_content))
        uploaded_image.verify()  # Verify the image format
        uploaded_image = Image.open(io.BytesIO(file_content))  # Reopen after verification

        # Set similarity threshold
        similarity_threshold = 0.1 

        # Retrieve and process image BLOBs from the database
        database_features = []
        database_image_blobs = []
        database_colors = []
        database_names = []
        database_prediction = []
        database_dateOfCaptured = []
        database_dog_age = []
        database_date_sex=[]


      

        for row in data1:
            image_blob = row[11]
            color = row[9]
            name = row[8]
            prediction = row[13]
            dateOfCaptured= row[3]
            dog_age = row[10]
            date_sex= row[14]
        
  
            try:
                encoded_image_blob = base64.b64encode(image_blob).decode('utf-8')
                binary_data = base64.b64decode(image_blob)
                
                img_image = Image.open(io.BytesIO(binary_data))

                # Extract features for each image
                features = extract_features(img_image, base_model)
                if features is not None and features.size > 0:
                    database_features.append(features)
                    database_image_blobs.append(binary_data)
                    database_colors.append(color)
                    database_names.append(name)
                    database_prediction.append(prediction)
                    database_dateOfCaptured.append(dateOfCaptured)
                    database_dog_age.append(dog_age)
                    database_date_sex.append(date_sex)
            
                        

                else:
                    print(f"Empty features for database image")
            except Exception as e:
                print(f"Error processing database image: {e}")

        if not database_features:
            return jsonify({'error': 'No valid images found in the database'}), 500

        # Extract features from the uploaded image
        uploaded_features = extract_features(uploaded_image, base_model)

        # Calculate similarities
        similarities = [cosine_similarity(uploaded_features, db_feat).flatten()[0] for db_feat in database_features]

        if not similarities:
            return jsonify({'error': 'No similarities calculated'}), 500

        # Filter out scores below the similarity threshold
        filtered_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]

        if not filtered_indices:
            return jsonify({'error': 'No similar images found with score above threshold'}), 404

        # Find the indices of the top 5 most similar images that meet the threshold
        top_n_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)[:5]

        similar_images = []
        for index in top_n_indices:
            similar_images.append({
                'image_blob': base64.b64encode(database_image_blobs[index]).decode('utf-8'),
                'similarity_score': float(similarities[index]),
                'color': database_colors[index],
                'name': database_names[index],
                'prediction': database_prediction[index],
                'date_missing' : database_dateOfCaptured [index],
                'age': database_dog_age[index],
                'sex':  database_date_sex[index],
              
         
            })

        return jsonify({'similar_images': similar_images})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/findincident', methods=['POST'])
def look_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_content = file.read()
        uploaded_image = Image.open(io.BytesIO(file_content))
        uploaded_image.verify()  # Verify the image format
        uploaded_image = Image.open(io.BytesIO(file_content))  # Reopen after verification

        # Set similarity threshold
        similarity_threshold = 0.1 

        # Retrieve and process image BLOBs from the database
        database_features = []
        database_image_blobs = []
        database_description = []
        database_place_names = []
    

        for row in data2:       
            image_blob = row[5]
            description = row[1]
            place_name = row[2]
            
            try:
                encoded_image_blob = base64.b64encode(image_blob).decode('utf-8')
                binary_data = base64.b64decode(encoded_image_blob)
                
                img_image = Image.open(io.BytesIO(binary_data))

                # Extract features for each image
                features = extract_features(img_image, base_model)
                if features is not None and features.size > 0:
                    database_features.append(features)
                    database_image_blobs.append(binary_data)
                    database_description.append(description)
                    database_place_names.append(place_name)
                   
                else:
                    print(f"Empty features for database image")
            except Exception as e:
                print(f"Error processing database image: {e}")

        if not database_features:
            return jsonify({'error': 'No valid images found in the database'}), 500

        # Extract features from the uploaded image
        uploaded_features = extract_features(uploaded_image, base_model)

        # Calculate similarities
        similarities = [cosine_similarity(uploaded_features, db_feat).flatten()[0] for db_feat in database_features]

        if not similarities:
            return jsonify({'error': 'No similarities calculated'}), 500

        # Filter out scores below the similarity threshold
        filtered_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]

        if not filtered_indices:
            return jsonify({'error': 'No similar images found with score above threshold'}), 404

        # Find the indices of the top 5 most similar images that meet the threshold
        top_n_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)[:5]

        similar_images = []
        for index in top_n_indices:
            similar_images.append({
                'image': base64.b64encode(database_image_blobs[index]).decode('utf-8'),
                'similarity_score': float(similarities[index]),
                'description': database_description[index],
                'place_name': database_place_names[index],
             
            })

        return jsonify({'similar_images': similar_images})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 5000, app, use_debugger=True, use_reloader=True)
