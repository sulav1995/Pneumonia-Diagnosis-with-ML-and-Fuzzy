from flask import Flask, request, send_file, render_template, jsonify, session, redirect, url_for
from flask_cors import CORS

from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.valuerep import PersonName
from pydicom.filewriter import write_file_meta_info
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import logging
from datetime import datetime
import uuid
import pandas as pd
from scipy.stats import zscore
import skfuzzy as fuzz
from werkzeug.utils import secure_filename
import skfuzzy.control as ctrl
from sklearn.decomposition import PCA
import cv2
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64  # New import for encoding Grad-CAM
import zlib  # New import for compression

app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = 'your_secret_key'  # To enable session storage

# Set the maximum allowed payload (file size) to 32MB
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB file size limit

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def homepage():
    return render_template('homepage.html')

# Define PrimaryCaps and CapsuleLayer Classes
class PrimaryCaps(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(filters=self.num_capsules * self.dim_capsules,
                                           kernel_size=3,
                                           strides=1,
                                           padding='same')
        super(PrimaryCaps, self).build(input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        output = tf.reshape(output, (-1, output.shape[1] * output.shape[2] * self.num_capsules, self.dim_capsules))
        return self.squash(output)

    def squash(self, s):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * s

    def get_config(self):
        config = super(PrimaryCaps, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsules': self.dim_capsules
        })
        return config

class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(shape=[input_shape[-1], self.num_capsules * self.dim_capsules],
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        inputs_hat = tf.einsum('bij,jk->bik', inputs, self.W)
        inputs_hat = tf.reshape(inputs_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsules))
        b = tf.zeros_like(inputs_hat[:, :, :, 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = self.squash(tf.reduce_sum(c[:, :, :, tf.newaxis] * inputs_hat, axis=1))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * outputs[:, tf.newaxis, :, :], axis=-1)
        return outputs

    def squash(self, s):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * s

    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsules': self.dim_capsules,
            'routings': self.routings
        })
        return config

# Load models and enhancements
MODEL_PATH = 'D:/3rd sem Project/Programs/Website/Second_final_model.h5'
CSV_PATH = 'D:/3rd sem Project/Programs/Website/fuzzy_k_enhancements.csv'
custom_objects = {'PrimaryCaps': PrimaryCaps, 'CapsuleLayer': CapsuleLayer}
model = load_model(MODEL_PATH, custom_objects=custom_objects)
fuzzy_k_enhancements_df = pd.read_csv(CSV_PATH)

# Function to preprocess the image
def preprocess_image(file_path):
    # Open the image and preprocess it
    image = Image.open(file_path).convert('RGB')  # Open and convert to RGB
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    return np.expand_dims(image, axis=0)

# Function to compute Grad-CAM heatmap
def generate_grad_cam(model, img_tensor, last_conv_layer_name='conv2d_3', pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap to a range between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay the heatmap on the original image
def overlay_heatmap_on_image(img_path, heatmap, intensity=0.5, colormap=cv2.COLORMAP_JET):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))

    # Resize heatmap to match the image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply the heatmap to the image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Overlay the heatmap on the original image
    overlayed_image = cv2.addWeighted(image, 1 - intensity, heatmap, intensity, 0)

    # Save and return the path to the overlayed image in the static/uploads folder
    static_folder_path = os.path.join('static', 'uploads')
    if not os.path.exists(static_folder_path):
        os.makedirs(static_folder_path)

    heatmap_path = os.path.join(static_folder_path, 'gradcam_overlay.png')
    cv2.imwrite(heatmap_path, overlayed_image)
    
    return 'uploads/gradcam_overlay.png'

# Function to compress Grad-CAM base64 string
def encode_gradcam_base64_compressed(heatmap):
    # Convert heatmap to image
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap))
    buffered = BytesIO()
    heatmap_image.save(buffered, format="PNG")
    
    # Base64 encode the image
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Compress the base64 string using zlib
    compressed_data = zlib.compress(base64_data.encode('utf-8'))
    return compressed_data

# Function to decompress the Grad-CAM base64 string
def decode_gradcam_base64_compressed(compressed_data):
    # Decompress the stored data
    base64_data = zlib.decompress(compressed_data).decode('utf-8')
    return base64_data


# Fuzzy logic-based classification using actual user inputs
def fuzzy_logic_classification(breathlessness, sputum_production, fever_duration, fever_value, hemoptysis):
    # Normalize fever duration and value
    fever_duration_normalized = fever_duration / 30  # Normalize duration (assuming max 30 days)
    fever_value_normalized = (fever_value - 35) / 7  # Normalize fever value (assuming range 35°C to 42°C)

    # Create fuzzy input array
    condition_factors = np.array([breathlessness, sputum_production, fever_duration_normalized, fever_value_normalized, hemoptysis])
    return condition_factors


# Neuro Fuzzy Classification
def neuro_fuzzy_classification(fuzzy_logic_scores):
    # Ensuring fuzzy_logic_scores are scalar arrays
    fuzzy_k_scores = np.mean(fuzzy_k_scores, axis=0)
    fuzzy_logic_scores = np.mean(fuzzy_logic_scores, axis=0)
    final_score = fuzzy_logic_scores / 2
    return final_score

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            breathlessness = float(request.form['breathlessness'])
            sputum_production = float(request.form['sputum_production'])
            fever_duration = int(request.form['fever_duration'])
            fever_value = float(request.form['fever_value'])
            hemoptysis = float(request.form['hemoptysis'])

            preprocessed_image = preprocess_image(file_path)

            # Perform CNN prediction
            cnn_prediction = model.predict(preprocessed_image).flatten()[0]
            cnn_accuracy = float(cnn_prediction * 100)  # Treating the probability as accuracy percentage

            # Generate Grad-CAM for the prediction
            heatmap = generate_grad_cam(model, preprocessed_image)
            heatmap_path = overlay_heatmap_on_image(file_path, heatmap)

            # Compress and store the Grad-CAM heatmap as base64 in session
            compressed_gradcam = encode_gradcam_base64_compressed(heatmap)
            session['compressed_gradcam'] = compressed_gradcam  # Save compressed Grad-CAM in session

            # Apply fuzzy logic
            fuzzy_logic_scores = fuzzy_logic_classification(breathlessness, sputum_production, fever_duration, fever_value, hemoptysis)
            final_neuro_fuzzy_score = np.mean(fuzzy_logic_scores)  # Get final score from fuzzy logic

            severity = 'Negligible 'if final_neuro_fuzzy_score < 0.13 else 'Low Pneumonia' if final_neuro_fuzzy_score < 0.33 else 'Mild Pneumonia' if final_neuro_fuzzy_score < 0.66 else 'Severe Pneumonia'

            # Store fuzzy logic results in session
            session['fuzzy_logic_diagnosis'] = severity
            session['fuzzy_logic_scores'] = {
                'breathlessness': breathlessness,
                'sputum_production': sputum_production,
                'fever_duration': fever_duration,
                'fever_value': fever_value,
                'hemoptysis': hemoptysis,
                'severity': severity
            }

            # Store CNN accuracy and fuzzy-K accuracy in session
            session['cnn_accuracy'] = cnn_accuracy
            session['fuzzy_k_accuracy'] = final_neuro_fuzzy_score * 100

            # Pass prediction and accuracy values to the classifier template
            return render_template('classifier.html', prediction=f'Pneumonia ({severity})', 
                                   cnn_accuracy=cnn_accuracy, fuzzy_k_accuracy=final_neuro_fuzzy_score * 100, 
                                   gradcam_path=heatmap_path)

    return render_template('classifier.html')


@app.route('/show_main')
def show_main():
    # Retrieve the fuzzy logic diagnosis and metrics from the session
    fuzzy_logic_diagnosis = session.get('fuzzy_logic_diagnosis', 'Not Available')
    fuzzy_logic_scores = session.get('fuzzy_logic_scores', {})
    
    # Retrieve CNN and Fuzzy-K accuracy from the session
    cnn_accuracy = session.get('cnn_accuracy', 'Not Available')
    fuzzy_k_accuracy = session.get('fuzzy_k_accuracy', 'Not Available')
    
    # Retrieve compressed Grad-CAM from session and decompress it
    compressed_gradcam = session.get('compressed_gradcam', None)
    if compressed_gradcam:
        decompressed_gradcam = decode_gradcam_base64_compressed(compressed_gradcam)
    else:
        decompressed_gradcam = 'Not Available'
    
    return render_template('main.html', cnn_accuracy=cnn_accuracy, fuzzy_k_accuracy=fuzzy_k_accuracy, fuzzy_logic_diagnosis=fuzzy_logic_diagnosis, fuzzy_logic_scores=fuzzy_logic_scores, compressed_gradcam=decompressed_gradcam)


# Expanded Mapping Layer: Local codes to standardized codes
CODE_MAPPINGS = {
    'modality': {
        'LOCAL_MR': 'MR',  # Valid modality code for Magnetic Resonance
        'LOCAL_CT': 'CT',  # Valid modality code for Computed Tomography
    },
    'bodySite': {
        'LOCAL_CHEST': '51185008',  # SNOMED code for Chest
        'LOCAL_HEAD': '69536005',    # SNOMED code for Head
    },
    'sopClass': {
        'LOCAL_MR_IMAGE': '110181',  # SOP Class UID code for MR Image
        'LOCAL_CT_IMAGE': '110181',  # SOP Class UID code for CT Image
    },
    'patientSex': {
        'M': 'male',
        'F': 'female',
        'O': 'other',
    },
    'maritalStatus': {
        'S': 'S',  # Single -> Never Married
        'M': 'M',  # Married
        'W': 'W',  # Widowed
        'D': 'D',  # Divorced
    },
    'contactRelationship': {
        'PRN': 'FAMMEMB',  # Family Member
    }
}

def format_date(date_str):
    return date_str.replace('-', '')

def format_time(time_str):
    return time_str.replace(':', '') + "00"


@app.route('/accuracy_main')
def accuracy_main():
    cnn_accuracy = session.get('cnn_accuracy', 0)
    fuzzy_k_accuracy = session.get('fuzzy_k_accuracy', 0)
    compressed_gradcam = session.get('compressed_gradcam', '')

    return render_template('main.html', cnn_accuracy=cnn_accuracy, fuzzy_k_accuracy=fuzzy_k_accuracy, compressed_gradcam=compressed_gradcam)

@app.route('/dicom_to_fhir')
def dicom_to_fhir():
    return render_template('dicom_to_fhir.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/pneumonia_time_series')
def pneumonia_time_series():
    return render_template('Pneumonia_time_series.html')

@app.route('/dicom_reader')
def dicom_reader():
    return render_template('dicom_reader.html')

@app.route('/create_dicom', methods=['POST'])
def create_dicom():
    try:
        logging.debug("Processing request to create DICOM file.")

        # Ensure file upload is handled properly
        if 'imageUpload' not in request.files:
            logging.error("No image file part in the request.")
            return "No image file part in the request.", 400

        file = request.files['imageUpload']
        if file.filename == '':
            logging.error("No selected file.")
            return "No selected file.", 400
        
        logging.debug(f"Image file received: {file.filename}")
        
        # Capture metadata from the form
        metadata = {key: request.form[key] for key in request.form}
        logging.debug(f"Metadata received: {metadata}")

        # Capture CNN and Fuzzy K accuracy
        cnn_accuracy = metadata.get('cnn_accuracy', None)
        fuzzy_k_accuracy = metadata.get('fuzzy_k_accuracy', None)
        compressed_gradcam = session.get('compressed_gradcam', None)  # Retrieve Grad-CAM from session

        # Capture fuzzy logic inputs for symptoms
        fuzzy_logic_diagnosis = session.get('fuzzy_logic_diagnosis', None)  # Fuzzy logic diagnosis
        fuzzy_logic_scores = session.get('fuzzy_logic_scores', None)  # Fuzzy logic score dict for symptoms

        # Load the JPEG image
        image = Image.open(file)
        image = np.array(image)
        logging.debug(f"Image loaded and converted to numpy array. Shape: {image.shape}")

        # Create a DICOM dataset
        file_meta = Dataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = metadata['sopClassUID']
        file_meta.MediaStorageSOPInstanceUID = metadata['sopInstanceUID']
        file_meta.ImplementationClassUID = "1.2.3.4"

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = metadata['sopClassUID']
        ds.SOPInstanceUID = metadata['sopInstanceUID']
        ds.StudyDate = format_date(metadata['studyDate'])
        ds.SeriesDate = format_date(metadata['seriesDate'])
        ds.AcquisitionDate = format_date(metadata['acquisitionDate'])
        ds.ContentDate = format_date(metadata['contentDate'])
        ds.StudyTime = format_time(metadata['studyTime'])
        ds.SeriesTime = format_time(metadata['seriesTime'])
        ds.AcquisitionTime = format_time(metadata['acquisitionTime'])
        ds.ContentTime = format_time(metadata['contentTime'])
        ds.Modality = metadata['modality'].upper()
        ds.Manufacturer = metadata['manufacturer']

        algo_ds = Dataset()
        algo_ds.CodeValue = "algorithm"
        algo_ds.CodingSchemeDesignator = "algo"
        algo_ds.CodeMeaning = metadata['algorithmNameCodeSequence']
        ds.AlgorithmNameCodeSequence = [algo_ds]

        ds.PatientName = PersonName(metadata['patientName'])
        ds.PatientID = metadata['patientID']
        ds.PatientBirthDate = format_date(metadata['patientBirthDate'])
        ds.PatientSex = metadata['patientSex'][0]
        ds.StudyInstanceUID = metadata['studyInstanceUID']
        ds.SeriesInstanceUID = metadata['seriesInstanceUID']
        
        ds.PatientAge = metadata['patientAge'] + 'Y'
        ds.PatientWeight = metadata['patientWeight']
        ds.PatientAddress = metadata['patientAddress']
        ds.ReferringPhysicianName = metadata['referringPhysicianName']
        ds.InstitutionName = metadata['institutionName']
        ds.StationName = metadata['stationName']
        ds.StudyDescription = metadata['studyDescription']
        ds.SeriesDescription = metadata['seriesDescription']
        ds.ManufacturerModelName = metadata['manufacturerModelName']
        ds.SoftwareVersions = metadata['softwareVersions']

        # New fields
        ds.MaritalStatus = metadata['maritalStatus']
        ds.ContactRelationship = metadata['contactRelationship']
        ds.ContactGender = metadata['contactGender']
        ds.CommunicationLanguage = metadata['communicationLanguage']
        ds.LinkType = metadata['linkType']

        logging.debug("DICOM dataset populated with metadata.")

        # Declare a private creator for your custom tags
        ds.add_new((0x0029, 0x0010), 'LO', 'Diagnosis Information')

        # Add CNN and Fuzzy K Accuracy to DICOM
        if cnn_accuracy is not None:
            ds.add_new((0x0029, 0x1000), 'DS', str(cnn_accuracy))  # CNN Accuracy as Decimal String

        if fuzzy_k_accuracy is not None:
            ds.add_new((0x0029, 0x1001), 'DS', str(fuzzy_k_accuracy))  # Fuzzy K Accuracy as Decimal String

        # Add compressed Grad-CAM to DICOM
        if compressed_gradcam is not None:
            ds.add_new((0x0029, 0x1002), 'OB', compressed_gradcam)  # Store compressed Grad-CAM

        # Add fuzzy logic diagnosis and scores to DICOM
        if fuzzy_logic_diagnosis is not None:
            ds.add_new((0x0029, 0x1003), 'LO', fuzzy_logic_diagnosis)  # Store Fuzzy Logic diagnosis
            
        if fuzzy_logic_scores is not None:
            ds.add_new((0x0029, 0x1004), 'DS', str(fuzzy_logic_scores.get('breathlessness', 'N/A')))
            ds.add_new((0x0029, 0x1005), 'DS', str(fuzzy_logic_scores.get('sputum_production', 'N/A')))
            ds.add_new((0x0029, 0x1006), 'DS', str(fuzzy_logic_scores.get('fever_duration', 'N/A')))
            ds.add_new((0x0029, 0x1007), 'DS', str(fuzzy_logic_scores.get('fever_value', 'N/A')))
            ds.add_new((0x0029, 0x1008), 'DS', str(fuzzy_logic_scores.get('hemoptysis', 'N/A')))

        # Image data
        ds.Rows, ds.Columns = image.shape[:2]
        ds.SamplesPerPixel = 1 if len(image.shape) == 2 else image.shape[2]
        ds.PhotometricInterpretation = "MONOCHROME2" if ds.SamplesPerPixel == 1 else "RGB"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = image.tobytes()
        logging.debug("DICOM dataset populated with image data.")
        
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        dicom_path = os.path.join(tempfile.gettempdir(), 'output.dcm')
        ds.save_as(dicom_path)
        logging.debug(f"DICOM file saved to: {dicom_path}")

        return send_file(dicom_path, as_attachment=True, download_name='output.dcm')
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return str(e), 500

@app.route('/convert_to_fhir', methods=['POST'])
def convert_to_fhir():
    try:
        # Save and read the DICOM file
        file = request.files['dicomFile']
        dicom_path = os.path.join(tempfile.gettempdir(), 'uploaded.dcm')
        file.save(dicom_path)
        ds = dcmread(dicom_path, force=True)

        # Extract metadata from the DICOM file
        metadata = {attr: getattr(ds, attr, 'N/A') for attr in [
            'TransferSyntaxUID', 'SOPClassUID', 'SOPInstanceUID', 'StudyDate',
            'SeriesDate', 'AcquisitionDate', 'ContentDate', 'StudyTime',
            'SeriesTime', 'AcquisitionTime', 'ContentTime', 'Modality',
            'Manufacturer', 'PatientName', 'PatientID', 'PatientBirthDate', 
            'PatientSex', 'StudyInstanceUID', 'SeriesInstanceUID', 'PatientAge',
            'PatientWeight', 'PatientAddress', 'ReferringPhysicianName', 
            'InstitutionName', 'StationName', 'StudyDescription', 
            'SeriesDescription', 'ManufacturerModelName', 'SoftwareVersions'
        ]}

        # Extract CNN and Fuzzy K accuracy from private DICOM tags
        cnn_accuracy = ds.get((0x0029, 0x1000), 'N/A')
        fuzzy_k_accuracy = ds.get((0x0029, 0x1001), 'N/A')

        # Convert the extracted values to the correct format
        cnn_accuracy = cnn_accuracy.value if cnn_accuracy != 'N/A' else 'N/A'
        fuzzy_k_accuracy = fuzzy_k_accuracy.value if fuzzy_k_accuracy != 'N/A' else 'N/A'

        # Function to map local codes to standardized codes
        def map_code(category, value):
            return CODE_MAPPINGS.get(category, {}).get(value, value)

        # Apply mappings
        metadata['Modality'] = map_code('modality', metadata['Modality'])
        metadata['SOPClassUID'] = map_code('sopClass', metadata['SOPClassUID'])
        metadata['PatientSex'] = map_code('patientSex', metadata['PatientSex'])
        metadata['BodySite'] = map_code('bodySite', 'LOCAL_CHEST')  # Assuming 'LOCAL_CHEST' is provided

        # Base FHIR URL
        base_url = "http://hl7.org/fhir"

        # Correct the date format to YYYY-MM-DD
        def format_date(date_str):
            try:
                return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            except ValueError:
                return 'N/A'

        # Handle Patient Name
        if isinstance(ds.PatientName, PersonName):
            patient_name_family = ds.PatientName.family_name or 'Unknown'
            patient_name_given = ds.PatientName.given_name or 'Unknown'
        else:
            patient_name_family = 'Unknown'
            patient_name_given = 'Unknown'

        # Handle Referring Physician Name
        referring_physician_name = str(ds.ReferringPhysicianName) if isinstance(ds.ReferringPhysicianName, PersonName) else 'Unknown'

        # Generate unique resource IDs
        imaging_study_uuid = uuid.uuid4().hex
        patient_uuid = uuid.uuid4().hex
        document_reference_uuid = uuid.uuid4().hex

        # FHIR DocumentReference resource (without Grad-CAM pixel data)
        document_reference = {
            "resourceType": "DocumentReference",
            "id": document_reference_uuid,
            "status": "current",
            "subject": {
                "reference": f"{base_url}/Patient/{patient_uuid}"
            },
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "18748-4",  # Valid code for diagnostic imaging study
                        "display": "Diagnostic imaging study"
                    }
                ]
            },
            "content": [
                {
                    "attachment": {
                        "contentType": "image/jpeg",
                        "title": "Imaging Study Attachment",
                        "creation": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "url": "http://127.0.0.1:5000/classifier/dicom-image"  # If applicable
                    }
                }
            ],
            "text": {
                "status": "generated",
                "div": "<div xmlns='http://www.w3.org/1999/xhtml'><p>Document reference for imaging study.</p></div>"
            }
        }

        # Correct gender code (M/F) mapping to 'male'/'female'
        gender_map = {'M': 'male', 'F': 'female', 'O': 'other', 'U': 'unknown'}
        patient_gender = gender_map.get(metadata['PatientSex'].upper(), 'unknown')

        # FHIR Bundle with ImagingStudy, Patient, and DocumentReference
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "fullUrl": f"{base_url}/ImagingStudy/{imaging_study_uuid}",
                    "resource": {
                        "resourceType": "ImagingStudy",
                        "id": imaging_study_uuid,
                        "status": "available",
                        "subject": {
                            "reference": f"{base_url}/Patient/{patient_uuid}"
                        },
                        "started": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "modality": [
                            {
                                "system": "http://dicom.nema.org/resources/ontology/DCM",
                                "code": metadata['Modality'],
                                "display": "Magnetic Resonance"  # Example, should be dynamic
                            }
                        ],
                        "description": metadata['StudyDescription'],
                        "series": [
                            {
                                "uid": metadata['SeriesInstanceUID'],
                                "number": 1,
                                "modality": {
                                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                                    "code": metadata['Modality']
                                },
                                "description": metadata['SeriesDescription'],
                                "numberOfInstances": 1,
                                "bodySite": {
                                    "system": "http://snomed.info/sct",
                                    "code": metadata['BodySite'],
                                    "display": "Chest"
                                },
                                "instance": [
                                    {
                                        "uid": metadata['SOPInstanceUID'],
                                        "sopClass": {
                                            "code": metadata['SOPClassUID'],
                                            "system": "http://dicom.nema.org/resources/ontology/DCM",
                                            "display": "SOP Class UID"
                                        },
                                        "number": 1,
                                        "title": "Instance Title"
                                    }
                                ],
                                # Include CNN and Fuzzy-K accuracy as part of the series
                                "extension": [
                                    {
                                        "url": "http://127.0.0.1:5000/classifier/cnn-accuracy",
                                        "valueDecimal": float(cnn_accuracy) if cnn_accuracy != 'N/A' else None
                                    },
                                    {
                                        "url": "http://127.0.0.1:5000/classifier/fuzzy-k-accuracy",
                                        "valueDecimal": float(fuzzy_k_accuracy) if fuzzy_k_accuracy != 'N/A' else None
                                    }
                                ]
                            }
                        ],
                        "text": {
                            "status": "generated",
                            "div": "<div xmlns='http://www.w3.org/1999/xhtml'><p>Imaging study for chest MRI with CNN and Fuzzy K accuracy.</p></div>"
                        }
                    }
                },
                {
                    "fullUrl": f"{base_url}/Patient/{patient_uuid}",
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_uuid,
                        "text": {
                            "status": "generated",
                            "div": f"<div xmlns='http://www.w3.org/1999/xhtml'><p>Patient: {patient_name_given} {patient_name_family}, Gender: {patient_gender}</p></div>"
                        },
                        "name": [
                            {
                                "use": "official",
                                "family": patient_name_family,
                                "given": [patient_name_given]
                            }
                        ],
                        "gender": patient_gender,  # Corrected gender mapping
                        "birthDate": format_date(metadata['PatientBirthDate']),  # Corrected birth date format
                    }
                },
                {
                    "fullUrl": f"{base_url}/DocumentReference/{document_reference_uuid}",
                    "resource": document_reference
                }
            ]
        }

        # Return the FHIR resource as a JSON response
        return jsonify(fhir_bundle)

    except Exception as e:
        logging.error(f"Error converting DICOM to FHIR: {e}", exc_info=True)
        return str(e), 500

# Code expanded on 12th November

# Load model path for CNN (ensure model path is unique if not already defined)
CNN_MODEL_PATH = 'D:/3rd sem Project/Programs/Website/Third_final_model.h5'
custom_objects = {'PrimaryCaps': PrimaryCaps, 'CapsuleLayer': CapsuleLayer}  # Ensure custom layers are recognized
cnn_model = load_model(CNN_MODEL_PATH, custom_objects=custom_objects)

@app.route('/upload_xray', methods=['POST'])
def upload_xray():
    try:
        if 'xrayImage' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['xrayImage']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the X-ray image temporarily
        xray_path = os.path.join(tempfile.gettempdir(), 'uploaded_xray.png')
        file.save(xray_path)

        # Process the image with the CNN model
        confidence_score = process_xray_with_cnn(xray_path, cnn_model)

        # Check if the result is a scalar or an array
        if isinstance(confidence_score, np.ndarray):
            confidence_score = confidence_score.item()  # Convert single-item array to scalar

        # Convert confidence_score to a standard Python float if necessary
        confidence_score = float(confidence_score)

        # Determine the prediction based on the confidence score
        prediction = 'Pneumonia' if confidence_score >= 0.5 else 'Normal'

        return jsonify({
            'prediction': prediction,
            'confidence': confidence_score
        })

    except Exception as e:
        app.logger.error(f"Error in /upload_xray: {str(e)}")
        return jsonify({'error': 'Failed to process X-ray image.'}), 500

def process_xray_with_cnn(image_path, model):
    """Processes an X-ray image with the CNN model and returns the confidence score."""
    try:
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=(0, -1))

        predictions = model.predict(image_array)
        print("Predictions shape:", predictions.shape) 
        
        # Check the shape of predictions to determine the model's output format
        if predictions.shape[1] == 2:  # Binary classification with two outputs (e.g., [Normal, Pneumonia])
            pneumonia_confidence = predictions[0][1]  # Confidence for the Pneumonia class
        elif predictions.shape[1] == 1:  # Single output, likely a single confidence score
            pneumonia_confidence = predictions[0][0]  # Use the single output as the confidence score
        else:
            app.logger.error("Unexpected output shape from model predictions")
            return None
            
        return pneumonia_confidence
    except Exception as e:
        app.logger.error(f"Error in process_xray_with_cnn: {str(e)}")
        return None

# Define route for DICOM upload
@app.route('/upload_dicom', methods=['POST'])
def upload_dicom():
    try:
        # Ensure the request has a DICOM file
        if 'dicomFile' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['dicomFile']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the DICOM file as patient_followup.dcm in UPLOAD_FOLDER
        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], 'patient_followup.dcm')
        file.save(dicom_path)
        
        # Attempt to read the DICOM file to confirm it's valid
        try:
            ds = dcmread(dicom_path)
        except Exception as e:
            return jsonify({'error': f'Failed to read DICOM file: {str(e)}'}), 500
        
        # Extract relevant DICOM data
        data = {
            'diagnosis': ds.get((0x0029, 0x1003), 'N/A').value if (0x0029, 0x1003) in ds else 'N/A',
            'breathlessness': ds.get((0x0029, 0x1004), 'N/A').value if (0x0029, 0x1004) in ds else 'N/A',
            'sputum_production': ds.get((0x0029, 0x1005), 'N/A').value if (0x0029, 0x1005) in ds else 'N/A',
            'fever_duration': ds.get((0x0029, 0x1006), 'N/A').value if (0x0029, 0x1006) in ds else 'N/A',
            'fever_value': ds.get((0x0029, 0x1007), 'N/A').value if (0x0029, 0x1007) in ds else 'N/A',
            'hemoptysis': ds.get((0x0029, 0x1008), 'N/A').value if (0x0029, 0x1008) in ds else 'N/A',
            'content_date': ds.get((0x0008, 0x0023), 'N/A').value if (0x0008, 0x0023) in ds else 'N/A'
        }

        # Print the extracted content_date for debugging
        print("Extracted content_date from DICOM:", data['content_date'])

        # Convert content_date from YYYYMMDD to MM/DD/YYYY
        if data['content_date'] != 'N/A' and len(data['content_date']) == 8:
            try:
                date_obj = datetime.strptime(data['content_date'], "%Y%m%d")
                data['content_date'] = date_obj.strftime("%m/%d/%Y")
                # Print the converted content_date for debugging
                print("Converted content_date to MM/DD/YYYY:", data['content_date'])
            except ValueError:
                data['content_date'] = 'Invalid Date'  # Handle incorrect date formats gracefully

        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Define the path for storing symptom trends
SYMPTOM_TREND_PATH = os.path.join(UPLOAD_FOLDER, 'symptom_trend.csv')

# Enable debugging level logging
app.logger.setLevel(logging.DEBUG)

# Functions to save/load symptom trends
def save_symptom_trends(symptom_data):
    """Saves symptom trends in a CSV file for future loading"""
    if not os.path.exists(SYMPTOM_TREND_PATH):
        df = pd.DataFrame(columns=['date', 'symptom', 'value'])
    else:
        df = pd.read_csv(SYMPTOM_TREND_PATH)
    symptom_data['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Create a new DataFrame from the symptom data to append
    new_data = pd.DataFrame([symptom_data])
    # Concatenate the new data with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(SYMPTOM_TREND_PATH, index=False)

def load_symptom_trends():
    """Loads saved symptom trends if available, or returns an empty DataFrame if not."""
    if os.path.exists(SYMPTOM_TREND_PATH):
        return pd.read_csv(SYMPTOM_TREND_PATH)
    return pd.DataFrame(columns=['date', 'symptom', 'value'])

# Enhanced Dynamic Fuzzy Membership Adjustment Function with Contextual Weighting
def adjust_fuzzy_membership_contextual(symptom, trends, default_range=(0, 1), min_val=0, max_val=1, base_smoothing=0.3):
    """ Adjust fuzzy membership based on symptom trends over time with severity-based smoothing and adaptive range """
    
    if symptom in trends.columns:
        # Remove NaN values and outliers using Z-score filtering
        symptom_data = trends[symptom].dropna().values
        if len(symptom_data) > 0:
            symptom_data = symptom_data[(np.abs(zscore(symptom_data)) < 3)]
            
            # Determine severity with variance and mean of the recent data
            recent_variance = np.var(symptom_data[-10:])  # Calculate variance for the last 10 data points
            recent_mean = np.mean(symptom_data[-10:]) if len(symptom_data) >= 10 else np.mean(symptom_data)
            
            # Adjust smoothing factor based on symptom trend severity
            smoothing_factor = base_smoothing + (recent_variance / 100)  # Increase smoothing factor with variance
            
            # Apply exponential smoothing for weighted trend analysis
            weighted_data = np.zeros_like(symptom_data, dtype=float)
            weighted_data[0] = symptom_data[0]
            for i in range(1, len(symptom_data)):
                weighted_data[i] = smoothing_factor * symptom_data[i] + (1 - smoothing_factor) * weighted_data[i - 1]
            
            # Adjust percentiles based on recent mean as an indicator of symptom severity
            lower_percentile = 10 if recent_mean < 0.5 else 15
            upper_percentile = 90 if recent_mean < 0.5 else 85
            
            # Calculate the adjusted range based on updated percentiles
            adjusted_min = np.percentile(weighted_data, lower_percentile)
            adjusted_max = np.percentile(weighted_data, upper_percentile)
            
            # Clip range to ensure it falls within specified min and max limits
            return np.clip(np.arange(adjusted_min, adjusted_max, 0.1), min_val, max_val)
    
    # Use default range if no valid trends data is available
    return np.arange(default_range[0], default_range[1], 0.1)

# Load trends to dynamically adjust membership functions if trends are available
trends_df = load_symptom_trends()

# Fuzzy variables and membership functions setup
breathlessness = ctrl.Antecedent(adjust_fuzzy_membership_contextual('breathlessness', trends_df), 'breathlessness')
sputum_production = ctrl.Antecedent(adjust_fuzzy_membership_contextual('sputum_production', trends_df), 'sputum_production')
fever_duration = ctrl.Antecedent(adjust_fuzzy_membership_contextual('fever_duration', trends_df, default_range=(0, 30)), 'fever_duration')
fever_value = ctrl.Antecedent(adjust_fuzzy_membership_contextual('fever_value', trends_df, default_range=(35, 42)), 'fever_value')
hemoptysis = ctrl.Antecedent(adjust_fuzzy_membership_contextual('hemoptysis', trends_df), 'hemoptysis')
oxygen_level = ctrl.Antecedent(adjust_fuzzy_membership_contextual('oxygen_level', trends_df, default_range=(0, 100)), 'oxygen_level')
cough_severity = ctrl.Antecedent(adjust_fuzzy_membership_contextual('cough_severity', trends_df), 'cough_severity')
chest_pain = ctrl.Antecedent(adjust_fuzzy_membership_contextual('chest_pain', trends_df), 'chest_pain')
fatigue = ctrl.Antecedent(adjust_fuzzy_membership_contextual('fatigue', trends_df), 'fatigue')
appetite_loss = ctrl.Antecedent(adjust_fuzzy_membership_contextual('appetite_loss', trends_df), 'appetite_loss')
confusion = ctrl.Antecedent(adjust_fuzzy_membership_contextual('confusion', trends_df), 'confusion')
pneumonia_severity = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'pneumonia_severity')

for var in [breathlessness, sputum_production, fever_duration, fever_value, hemoptysis, oxygen_level, cough_severity, chest_pain, fatigue, appetite_loss, confusion]:
    var.automf(names=['poor', 'average', 'good'])
pneumonia_severity.automf(names=['negligible', 'mild', 'severe'])

# Define fuzzy rules
rules = [
    ctrl.Rule(breathlessness['good'] | fever_value['good'] | oxygen_level['poor'], pneumonia_severity['severe']),
    ctrl.Rule(breathlessness['average'] & sputum_production['average'] & fever_duration['average'], pneumonia_severity['mild']),
    ctrl.Rule(breathlessness['poor'] & sputum_production['poor'] & fever_duration['poor'] & oxygen_level['good'], pneumonia_severity['negligible']),
    ctrl.Rule(fatigue['good'] | cough_severity['good'] | chest_pain['good'], pneumonia_severity['mild']),
    ctrl.Rule(confusion['good'] & appetite_loss['good'] & hemoptysis['good'], pneumonia_severity['severe']),
    ctrl.Rule(fever_value['poor'] & oxygen_level['good'], pneumonia_severity['negligible']),
    ctrl.Rule(fever_value['average'] & fatigue['average'] & chest_pain['average'], pneumonia_severity['mild']),
    ctrl.Rule(fever_value['good'] & breathlessness['good'] & sputum_production['good'], pneumonia_severity['severe']),
    ctrl.Rule(oxygen_level['poor'] & confusion['good'], pneumonia_severity['severe']),
]

pneumonia_control_system = ctrl.ControlSystem(rules)
pneumonia_simulation = ctrl.ControlSystemSimulation(pneumonia_control_system)

# Function to generate clinician-friendly explanation with detailed debugging
def generate_clinician_explanation(symptoms, activated_rules):
    explanation_text = "Detailed Rule Activation Explanation for Clinicians:\n"
    rule_descriptions = [
        ("Rule 1: High breathlessness, high fever value, or low oxygen level suggests severe pneumonia.", 
         symptoms['breathlessness'] > 0.5 or symptoms['fever_value'] > 39 or symptoms['oxygen_level'] < 60),
        ("Rule 2: Moderate levels of breathlessness, sputum production, and fever duration suggests mild pneumonia.",
         0.3 <= symptoms['breathlessness'] <= 0.6 and 0.3 <= symptoms['sputum_production'] <= 0.6 and 1 <= symptoms['fever_duration'] <= 15),
        ("Rule 3: Low levels of breathlessness, sputum production, and fever duration with high oxygen level indicates negligible pneumonia.",
         symptoms['breathlessness'] < 0.3 and symptoms['sputum_production'] < 0.3 and symptoms['fever_duration'] < 5 and symptoms['oxygen_level'] > 80),
        ("Rule 4: High fatigue, cough severity, or chest pain levels indicate mild pneumonia.",
         symptoms['fatigue'] > 0.5 or symptoms['cough_severity'] > 0.5 or symptoms['chest_pain'] > 0.5),
        ("Rule 5: High levels of confusion, appetite loss, and hemoptysis suggest severe pneumonia.",
         symptoms['confusion'] > 0.5 and symptoms['appetite_loss'] > 0.5 and symptoms['hemoptysis'] > 0.5),
        ("Rule 6: Low fever value and high oxygen level indicate negligible pneumonia.",
         symptoms['fever_value'] < 37 and symptoms['oxygen_level'] > 85),
        ("Rule 7: Moderate fever value, fatigue, and chest pain suggest mild pneumonia.",
         0.4 <= symptoms['fever_value'] <= 39 and 0.4 <= symptoms['fatigue'] <= 0.6 and 0.4 <= symptoms['chest_pain'] <= 0.6),
        ("Rule 8: High fever value, breathlessness, and sputum production indicate severe pneumonia.",
         symptoms['fever_value'] > 39 and symptoms['breathlessness'] > 0.5 and symptoms['sputum_production'] > 0.5),
        ("Rule 9: Low oxygen level and high confusion indicate severe pneumonia.",
         symptoms['oxygen_level'] < 60 and symptoms['confusion'] > 0.5),
    ]
    
    # Iterate through rule descriptions and check if each condition is met
    for i, (description, condition_met) in enumerate(rule_descriptions, start=1):
        print(f"Evaluating {description}: Condition Met - {condition_met}")  # Log condition checks
        if condition_met:
            explanation_text += f"\n- {description} (Activated: Rule {i})"
    
    # Final check to see if explanation_text contains activated rules
    if explanation_text == "Detailed Rule Activation Explanation for Clinicians:\n":
        explanation_text += "\nNo specific rules were activated based on the provided symptoms."
    
    print("Generated Clinician Explanation (after evaluation):", explanation_text)
    return explanation_text.strip()

# Define function to calculate severity with customizable explanations
def calculate_fuzzy_severity_with_explanation(symptoms, custom_symptoms, cnn_confidence=None):
    try:
        # Reset the control system simulation inputs for each standard symptom
        pneumonia_simulation.inputs = {}       

        # Set standard inputs in the fuzzy simulation
        for key, value in symptoms.items():
            pneumonia_simulation.input[key] = value
            
        # Incorporate CNN confidence score as a fuzzy input if provided
        if cnn_confidence is not None:
            pneumonia_simulation.input['cnn_confidence'] = cnn_confidence
            print(f"{pneumonia_simulation} value at 979")

        # Track activated rules for explanation
        triggered_symptoms = {key: value for key, value in symptoms.items() if value > 0}
        activated_rules = []
        

        # Process each custom symptom by creating a temporary control system
        for custom_symptom in custom_symptoms:
            name = custom_symptom['name']
            impact = custom_symptom['impact']
            value = custom_symptom.get('value', 0)
            
            # Map custom symptom impacts to compatible membership levels
            if impact == 'average':
                impact = 'mild'
            elif impact == 'negligible':
                impact = 'poor'
            elif impact == 'severe':
                impact = 'good'
            else:
                app.logger.error(f"Invalid impact level '{impact}' for symptom '{name}'")
                continue  # Skip if impact is invalid

            # Define a temporary fuzzy variable for the custom symptom
            new_symptom = ctrl.Antecedent(np.arange(0, 1.1, 0.1), name)
            new_symptom.automf(names=['poor', 'mild', 'good'])

            # Create a temporary control system and simulation for this custom symptom
            custom_rule = ctrl.Rule(new_symptom[impact], pneumonia_severity[impact])
            custom_control_system = ctrl.ControlSystem([custom_rule])
            custom_simulation = ctrl.ControlSystemSimulation(custom_control_system)

            # Set the input for the custom symptom and compute its impact on severity
            custom_simulation.input[name] = value
            custom_simulation.compute()

            # Log activated rule for explanation
            if value > 0:
                triggered_symptoms[name] = value
                activated_rules.append(f"Custom rule for {name} with impact '{impact}' applied.")

         # Since custom severity cannot directly influence `pneumonia_simulation`, we aggregate them conceptually
         # by using the computed severity for logging or reporting purposes as an approximate value.

        # Compute final severity based on standard symptoms after processing custom symptoms
        pneumonia_simulation.compute()
        severity = pneumonia_simulation.output['pneumonia_severity']

        # Manually simulate rule activation by comparing symptoms to each rule's condition
        for i, rule in enumerate(rules, start=1):
            # Check if rule conditions are met based on symptom values
            # Simulating "activation" manually since `evaluate` is unavailable
            conditions_met = False
            if rule == rules[0] and (symptoms['breathlessness'] > 0.5 or symptoms['fever_value'] > 39 or symptoms['oxygen_level'] < 60):
                conditions_met = True
            elif rule == rules[1] and (0.3 <= symptoms['breathlessness'] <= 0.6 and 0.3 <= symptoms['sputum_production'] <= 0.6 and 1 <= symptoms['fever_duration'] <= 15):
                conditions_met = True
            elif rule == rules[2] and (symptoms['breathlessness'] < 0.3 and symptoms['sputum_production'] < 0.3 and symptoms['fever_duration'] < 5 and symptoms['oxygen_level'] > 80):
                conditions_met = True
            elif rule == rules[3] and (symptoms['fatigue'] > 0.5 or symptoms['cough_severity'] > 0.5 or symptoms['chest_pain'] > 0.5):
                conditions_met = True
            elif rule == rules[4] and (symptoms['confusion'] > 0.5 and symptoms['appetite_loss'] > 0.5 and symptoms['hemoptysis'] > 0.5):
                conditions_met = True
            elif rule == rules[5] and (symptoms['fever_value'] < 37 and symptoms['oxygen_level'] > 85):
                conditions_met = True
            elif rule == rules[6] and (0.4 <= symptoms['fever_value'] <= 39 and 0.4 <= symptoms['fatigue'] <= 0.6 and 0.4 <= symptoms['chest_pain'] <= 0.6):
                conditions_met = True
            elif rule == rules[7] and (symptoms['fever_value'] > 39 and symptoms['breathlessness'] > 0.5 and symptoms['sputum_production'] > 0.5):
                conditions_met = True
            elif rule == rules[8] and (symptoms['oxygen_level'] < 60 and symptoms['confusion'] > 0.5):
                conditions_met = True
            # Add other rule condition checks similarly

            if conditions_met:
                activated_rules.append(f"Activated rule {i}")

        # Construct the explanation
        explanation = {
            "severity": severity,
            "triggered_symptoms": list(triggered_symptoms.items()), 
            "activated_rules": activated_rules
        }
        
        return severity, explanation
    
    except Exception as e:
        app.logger.error(f"Error in calculate_fuzzy_severity_with_explanation: {str(e)}")
        raise
        

    
def generate_radar_chart(symptoms):
    labels = ['Breathlessness', 'Sputum Production', 'Fever Duration', 'Fever Value', 'Hemoptysis']
    symptom_values = [
        symptoms.get('breathlessness', 0),
        symptoms.get('sputum_production', 0),
        symptoms.get('fever_duration', 0) / 30,  # Normalize to [0, 1] if max is 30 days
        (symptoms.get('fever_value', 35) - 35) / 7,  # Normalize assuming range [35, 42]
        symptoms.get('hemoptysis', 0)
    ]
    symptom_values.append(symptom_values[0])  # Close the radar plot

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, symptom_values, color='blue', alpha=0.25)
    ax.plot(angles, symptom_values, color='blue', linewidth=2) 
    ax.set_yticklabels([])  # Remove radial ticks

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='blue', fontsize=12, fontweight='bold')
    ax.set_title("Symptom Distribution", color="blue", fontsize=16, fontweight="bold")

    # Save the plot to a BytesIO object and encode as base64
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


# Define route for follow-up submission
@app.route('/submit_follow_up', methods=['POST'])
def submit_follow_up():
    try:
        # Log received form data for debugging
        app.logger.debug(f"Form Data Received: {request.form}")
        symptoms = {
            'breathlessness': float(request.form['breathlessness']),
            'sputum_production': float(request.form['sputumProduction']),
            'fever_duration': int(request.form['feverDuration']),
            'fever_value': float(request.form['feverValue']),
            'hemoptysis': float(request.form['hemoptysis']),
            'oxygen_level': float(request.form['oxygenLevel']),
            'cough_severity': float(request.form['coughSeverity']),
            'chest_pain': float(request.form['chestPain']),
            'fatigue': float(request.form['fatigue']),
            'appetite_loss': float(request.form['appetiteLoss']),
            'confusion': float(request.form['confusion'])
        }

        #parse custom symptoms and log for debugging
        custom_symptoms = json.loads(request.form.get('customSymptoms', '[]'))
        app.logger.debug(f"Custom Symptoms Received: {custom_symptoms}")
        
        #calculate severity
        severity, explanation = calculate_fuzzy_severity_with_explanation(symptoms, custom_symptoms)
        severity_category = 'Negligible' if severity < 0.2 else 'Mild' if severity < 0.4 else 'Moderate' if severity < 0.6 else 'Severe'

        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], 'patient_followup.dcm')
        if not os.path.exists(dicom_path):
            return jsonify({"error": "DICOM file not found"}), 400

        ds = dcmread(dicom_path)
        sr_dataset = Dataset()
        sr_dataset.ContentDate = datetime.now().strftime('%Y%m%d')
        sr_dataset.ContentTime = datetime.now().strftime('%H%M%S')
        sr_dataset.InstanceNumber = len(ds.get((0x0029, 0x1000), [])) + 1

        sr_dataset.add_new((0x0029, 0x1010), 'DS', str(severity))
        sr_dataset.add_new((0x0029, 0x1011), 'LO', severity_category)
        sr_dataset.add_new((0x0029, 0x1012), 'LT', json.dumps(explanation["activated_rules"]))

        prev_severity = ds.get((0x0029, 0x1009), None)
        progression_trend = f"Previous Severity: {prev_severity}, Current Severity: {severity}" if prev_severity else "No previous data available"
        sr_dataset.add_new((0x0029, 0x1014), 'LT', progression_trend)
        ds.add_new((0x0029, 0x100A), 'SQ', [sr_dataset])

        output_dicom_path = os.path.join(tempfile.gettempdir(), 'output_with_xai.dcm')
        ds.save_as(output_dicom_path)

        radar_chart = generate_radar_chart(symptoms)
        clinician_explanation = generate_clinician_explanation(symptoms, explanation["activated_rules"])
        
        # Save symptom trends to symptom_trend.csv
        for symptom, value in symptoms.items():
            save_symptom_trends({'symptom': symptom, 'value': value})
        
        # Generate clinician explanation and log it
        clinician_explanation = generate_clinician_explanation(symptoms, explanation["activated_rules"])
        print("Clinician Explanation Sent to Frontend:", clinician_explanation)

        return jsonify({
            'severity': severity,
            'severity_category': severity_category,
            'triggered_symptoms': explanation["activated_rules"],
            'activated_rules': explanation["activated_rules"],
            'progression_trend': progression_trend,
            'dicom_path': output_dicom_path,
            'radar_chart': radar_chart,
            'detailed_explanation': clinician_explanation
        })

    except KeyError as e:
        app.logger.error(f"Missing data: {str(e)}")
        return jsonify({'error': f'Missing data: {str(e)}'}), 400
    except ValueError as e:
        app.logger.error(f"Invalid data format: {str(e)}")
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Error in /submit_follow_up: {str(e)}")
        return jsonify({'error': 'Failed to process follow-up data.'}), 500


if __name__ == '__main__':
    # Print the URL of the Flask server
    print("MetaClassifyDICOM is running on http://127.0.0.1:5000/")
    # Run the Flask server
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
