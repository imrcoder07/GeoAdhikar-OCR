"""
Configuration module for FRA OCR System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.parent
    
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_APP = os.getenv('FLASK_APP', 'app.py')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/data/fra_ocr.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File Upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', str(BASE_DIR / 'app' / 'static' / 'uploads'))
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png,pdf,tiff,bmp').split(','))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'app.log'))
    
    # OCR Settings
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', 0.6))
    DEFAULT_LANGUAGE_HINTS = os.getenv('DEFAULT_LANGUAGE_HINTS', 'hi,en').split(',')
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))
    
    # Server
    PORT = int(os.getenv('PORT', 5000))
    
    # Field extraction patterns
    FIELD_PATTERNS = {
        'patta_id': [
            r'पट्टा\s*(?:सं(?:ख्या)?|नं(?:बर)?|ID|No\.?)\s*[-:]?\s*([A-Z0-9/\-]+)',
            r'Patta\s*(?:No|Number|ID)\s*[-:]?\s*([A-Z0-9/\-]+)',
            r'(?:FRA|IFR|CR|CFR)[/\-]?\d{4}[/\-]?[A-Z]{2}[/\-]?\d+'
        ],
        'holder_name': [
            r'नाम\s*[-:]?\s*([^\n]+)',
            r'Name\s*[-:]?\s*([^\n]+)',
            r'धारक\s*का\s*नाम\s*[-:]?\s*([^\n]+)'
        ],
        'village': [
            r'(?:गांव|ग्राम|Village)\s*[-:]?\s*([^\n,]+)',
            r'ग्राम\s*पंचायत\s*[-:]?\s*([^\n,]+)'
        ],
        'district': [
            r'(?:जिला|District)\s*[-:]?\s*([^\n,]+)',
            r'जनपद\s*[-:]?\s*([^\n,]+)'
        ],
        'area': [
            r'(\d+(?:\.\d+)?)\s*(?:हेक्टेयर|hectare|ha)',
            r'(\d+(?:\.\d+)?)\s*(?:एकड़|acre)',
            r'(\d+(?:\.\d+)?)\s*(?:वर्ग\s*मीटर|sq\.?\s*m|sqm)'
        ]
    }
    
    # State abbreviations
    STATE_ABBREVIATIONS = {
        'MP': 'Madhya Pradesh',
        'CG': 'Chhattisgarh',
        'JH': 'Jharkhand',
        'OD': 'Odisha',
        'MH': 'Maharashtra',
        'RJ': 'Rajasthan',
        'UP': 'Uttar Pradesh',
        'UK': 'Uttarakhand',
        'WB': 'West Bengal',
        'AS': 'Assam',
        'AP': 'Andhra Pradesh',
        'TS': 'Telangana',
        'KA': 'Karnataka',
        'KL': 'Kerala',
        'TN': 'Tamil Nadu',
        'GJ': 'Gujarat'
    }
    
    # Unit conversions to hectares
    AREA_CONVERSIONS = {
        'hectare': 1.0,
        'ha': 1.0,
        'acre': 0.4047,
        'bigha_mp': 0.2529,  # Madhya Pradesh
        'bigha_raj': 0.2529,  # Rajasthan (pucca)
        'bigha_wb': 0.1338,  # West Bengal
        'guntha': 0.0101,
        'sqm': 0.0001,
        'sq.m': 0.0001,
        'square_meter': 0.0001,
        'sqft': 0.0000092903,
        'sq.ft': 0.0000092903,
        'square_feet': 0.0000092903
    }
    
    # Claim type mappings
    CLAIM_TYPE_MAPPINGS = {
        'IFR': ['Individual', 'IFR', 'व्यक्तिगत', 'వ్యక్తిగత'],
        'CR': ['Community', 'CR', 'सामुदायिक', 'సమాజ'],
        'CFR': ['CFR', 'Community Forest', 'सामुदायिक वन', 'కమ్యూనిటీ అటవీ']
    }
    
    # Honorific titles to remove
    HONORIFICS = [
        'Sri', 'Shri', 'श्री', 'Smt', 'Smt.', 'श्रीमती', 'Kumari', 'कुमारी',
        'Mr', 'Mr.', 'Mrs', 'Mrs.', 'Ms', 'Ms.', 'Dr', 'Dr.', 'Prof', 'Prof.'
    ]

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    
# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
