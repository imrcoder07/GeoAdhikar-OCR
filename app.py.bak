"""
Main Flask application for FRA OCR System
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from loguru import logger

# Import our modules
from config.config import Config, config
from app.models.database import init_db, get_db, OCRDocument, ProcessingLog
from app.core.ocr_processor import GeminiOCRProcessor
from app.core.field_extractor import FieldExtractor

# Initialize Flask app
app = Flask(__name__, 
           template_folder='app/templates',
           static_folder='app/static')

# Load configuration
app.config.from_object(config[os.getenv('FLASK_ENV', 'development')])

# Enable CORS
CORS(app)

# Initialize database
init_db()

# Initialize processors
ocr_processor = None
field_extractor = FieldExtractor()

def init_ocr_processor():
    """Initialize OCR processor with error handling"""
    global ocr_processor
    try:
        ocr_processor = GeminiOCRProcessor()
        logger.info("OCR Processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {e}")
        ocr_processor = None

# Try to initialize OCR processor
init_ocr_processor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ocr_ready': ocr_processor is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process OCR"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Check OCR processor
        if not ocr_processor:
            init_ocr_processor()
            if not ocr_processor:
                return jsonify({'error': 'OCR service not available. Please check API key.'}), 503
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Get language hints from request
        language_hints = request.form.get('languages', 'hi,en').split(',')
        
        # Process with OCR
        start_time = time.time()
        
        try:
            # Extract text using Gemini
            ocr_result = ocr_processor.extract_text_from_image(filepath, language_hints)
            
            # Extract fields using regex
            raw_text = ocr_result.get('raw_text', '')
            extracted_fields = field_extractor.extract(raw_text)
            
            # Merge Gemini structured fields with regex extraction
            gemini_fields = ocr_result.get('structured_fields', {})
            for field_name, field_data in extracted_fields.items():
                # Use Gemini data if available and confident, else use regex
                if field_name in gemini_fields:
                    gemini_conf = gemini_fields[field_name].get('confidence', 0)
                    regex_conf = field_data.get('confidence', 0)
                    if gemini_conf > regex_conf:
                        extracted_fields[field_name].update(gemini_fields[field_name])
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Save to database
            db = next(get_db())
            doc = OCRDocument(
                file_name=filename,
                file_path=filepath,
                patta_id=extracted_fields.get('patta_id', {}).get('value'),
                holder_name=extracted_fields.get('holder_name', {}).get('value'),
                relative_name=extracted_fields.get('relative_name', {}).get('value'),
                relation_type=extracted_fields.get('relative_name', {}).get('relation_type'),
                village=extracted_fields.get('village', {}).get('value'),
                district=extracted_fields.get('district', {}).get('value'),
                state=extracted_fields.get('state', {}).get('value'),
                area_hectares=extracted_fields.get('area', {}).get('value'),
                claim_type=extracted_fields.get('claim_type', {}).get('value'),
                claim_status=extracted_fields.get('status', {}).get('value'),
                raw_ocr_text=raw_text,
                structured_data=json.dumps(extracted_fields),
                language_detected=','.join(ocr_result.get('language_detected', [])),
                ocr_confidence=ocr_result.get('quality_score', 0.5),
                processing_status='completed',
                processing_time_ms=processing_time,
                process_date=datetime.utcnow()
            )
            
            # Check if needs verification
            verification_reasons = []
            for field_name, field_data in extracted_fields.items():
                if field_data.get('confidence', 1.0) < 0.6:
                    verification_reasons.append(f"{field_name} has low confidence")
            
            if verification_reasons:
                doc.needs_verification = True
                doc.verification_reasons = json.dumps(verification_reasons)
            
            # Parse issue date if present
            issue_date_str = extracted_fields.get('issue_date', {}).get('value')
            if issue_date_str:
                try:
                    doc.issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')
                except:
                    pass
            
            db.add(doc)
            db.commit()
            
            # Log processing
            log_entry = ProcessingLog(
                document_id=doc.id,
                action='OCR_COMPLETED',
                details=f"Processed in {processing_time}ms"
            )
            db.add(log_entry)
            db.commit()
            
            # Prepare response
            response = {
                'success': True,
                'document_id': doc.id,
                'filename': filename,
                'processing_time_ms': processing_time,
                'extracted_fields': extracted_fields,
                'raw_text': raw_text[:500] + '...' if len(raw_text) > 500 else raw_text,
                'needs_verification': doc.needs_verification,
                'verification_reasons': verification_reasons
            }
            
            db.close()
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            
            # Save error to database
            db = next(get_db())
            doc = OCRDocument(
                file_name=filename,
                file_path=filepath,
                processing_status='failed',
                error_message=str(e)
            )
            db.add(doc)
            db.commit()
            db.close()
            
            return jsonify({
                'success': False,
                'error': f"OCR processing failed: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of processed documents"""
    try:
        db = next(get_db())
        documents = db.query(OCRDocument).order_by(OCRDocument.upload_date.desc()).limit(100).all()
        
        result = [doc.to_dict() for doc in documents]
        db.close()
        
        return jsonify({
            'success': True,
            'documents': result,
            'total': len(result)
        })
        
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/document/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get specific document details"""
    try:
        db = next(get_db())
        doc = db.query(OCRDocument).filter_by(id=doc_id).first()
        
        if not doc:
            db.close()
            return jsonify({'error': 'Document not found'}), 404
        
        result = doc.to_dict()
        result['structured_data'] = json.loads(doc.structured_data) if doc.structured_data else {}
        result['raw_text'] = doc.raw_ocr_text
        
        db.close()
        return jsonify({
            'success': True,
            'document': result
        })
        
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/<int:doc_id>', methods=['GET'])
def export_document(doc_id):
    """Export document as JSON"""
    try:
        db = next(get_db())
        doc = db.query(OCRDocument).filter_by(id=doc_id).first()
        
        if not doc:
            db.close()
            return jsonify({'error': 'Document not found'}), 404
        
        # Prepare export data
        export_data = {
            'doc_id': doc.id,
            'pages': 1,
            'extracted': {},
            'raw_full_text': doc.raw_ocr_text,
            'provenance': {
                'primary_ocr_provider': 'Gemini',
                'processing_date': doc.process_date.isoformat() if doc.process_date else None
            },
            'verification': {
                'needs_verification': doc.needs_verification,
                'reasons': json.loads(doc.verification_reasons) if doc.verification_reasons else []
            }
        }
        
        # Parse structured data
        if doc.structured_data:
            structured = json.loads(doc.structured_data)
            for field_name, field_data in structured.items():
                export_data['extracted'][field_name] = {
                    'value': field_data.get('value'),
                    'confidence': field_data.get('confidence', 0),
                    'source': field_data.get('source', 'Unknown'),
                    'raw_text_snippet': field_data.get('raw'),
                    'extraction_note': field_data.get('note')
                }
        
        db.close()
        return jsonify(export_data)
        
    except Exception as e:
        logger.error(f"Error exporting document {doc_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Setup logging
    logger.add(
        Config.LOG_FILE,
        rotation="100 MB",
        level=Config.LOG_LEVEL
    )
    
    logger.info("Starting FRA OCR System...")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.DEBUG
    )
