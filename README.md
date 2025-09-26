# 🌳 FRA OCR System - Forest Rights Act Document Scanner

A comprehensive OCR system for extracting and normalizing data from Indian Forest Rights Act (FRA) documents including pattas, claims, and certificates.

## 📋 Features

- **Multi-format Support**: Process JPG, PNG, PDF, TIFF documents
- **Multilingual OCR**: Supports Hindi, English, Odia, Telugu and other Indian languages
- **Smart Field Extraction**: Automatically extracts key fields like:
  - Patta ID/Number
  - Holder Name & Relative Names
  - Village, District, State
  - Land Area (with unit conversion)
  - Issue Date
  - Claim Type (IFR/CR/CFR)
  - Claim Status
- **Data Normalization**: 
  - Converts dates to ISO format
  - Normalizes area measurements to hectares
  - Handles regional variations (bigha, guntha, etc.)
  - Transliterates non-Latin text
- **Confidence Scoring**: Provides confidence levels for each extracted field
- **Web Interface**: Easy-to-use drag-and-drop interface
- **REST API**: Full API for integration with other systems
- **Database Storage**: SQLite database for storing OCR results

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))
- Windows/Linux/Mac OS

### Step 1: Clone or Download the Project

```bash
cd C:\Users\deban\projects\fra-ocr-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### Step 5: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📖 Usage Guide

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Drag and drop your FRA document or click to browse
3. Select the languages present in your document
4. Click "Process Document"
5. View extracted fields with confidence scores
6. Export results as JSON

### API Usage

#### Upload and Process Document
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@path/to/document.jpg" \
  -F "languages=hi,en"
```

#### Get Document List
```bash
curl http://localhost:5000/api/documents
```

#### Get Specific Document
```bash
curl http://localhost:5000/api/document/1
```

#### Export as JSON
```bash
curl http://localhost:5000/api/export/1
```

## 📁 Project Structure

```
fra-ocr-system/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── README.md              # This file
│
├── app/
│   ├── core/
│   │   ├── ocr_processor.py      # Gemini OCR integration
│   │   └── field_extractor.py    # Field extraction logic
│   │
│   ├── models/
│   │   └── database.py           # SQLAlchemy models
│   │
│   ├── utils/
│   │   └── normalization.py      # Data normalization functions
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Frontend styles
│   │   ├── js/
│   │   │   └── app.js            # Frontend JavaScript
│   │   └── uploads/              # Uploaded files
│   │
│   └── templates/
│       └── index.html            # Main HTML template
│
├── config/
│   └── config.py                 # Application configuration
│
├── data/                         # SQLite database location
├── logs/                         # Application logs
└── tests/                        # Test files
```

## 🔧 Configuration

Edit `config/config.py` to customize:

- **Field Patterns**: Regex patterns for field extraction
- **Area Conversions**: Regional unit conversions
- **State Abbreviations**: State name mappings
- **Language Settings**: Default language hints
- **Upload Limits**: File size and type restrictions

## 📊 Data Normalization Details

### Name Normalization
- Removes honorifics (Sri, Smt, etc.)
- Fixes common OCR errors
- Handles transliteration
- Standardizes capitalization

### Date Normalization
- Supports multiple input formats (DD/MM/YYYY, DD-MM-YYYY, etc.)
- Converts Hindi dates
- Outputs ISO format (YYYY-MM-DD)

### Area Normalization
- Converts various units to hectares:
  - Acres → Hectares (1 acre = 0.4047 ha)
  - Square meters → Hectares (10,000 m² = 1 ha)
  - Bigha → Hectares (varies by state)
  - Guntha → Hectares (1 guntha = 0.0101 ha)

### Location Normalization
- State abbreviation expansion (MP → Madhya Pradesh)
- District validation
- Village name fuzzy matching

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not found"**
   - Make sure you've created `.env` file from `.env.example`
   - Add your actual Gemini API key

2. **"No module named 'google.generativeai'"**
   - Run `pip install -r requirements.txt`

3. **"Port 5000 already in use"**
   - Change port in `.env` file: `PORT=5001`

4. **OCR not detecting text**
   - Ensure image is clear and well-lit
   - Try preprocessing (rotate, enhance contrast)
   - Check language hints match document language

## 📈 Performance Tips

1. **Image Quality**: Use 300 DPI scans for best results
2. **File Size**: Keep images under 10MB for faster processing
3. **Batch Processing**: Use the batch API for multiple documents
4. **Caching**: Enable Redis caching for production deployments

## 🚢 Production Deployment

### Deploy to Render

1. **Fork or clone this repository** to your GitHub account.

2. **Create a new Web Service on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the branch (main/master)

3. **Configure the service**:
   - **Runtime**: Docker (since Dockerfile is present)
   - **Dockerfile Path**: `./Dockerfile` (auto-detected)
   - The Dockerfile handles all system dependencies (Tesseract, OpenCV libs) and Python packages

4. **Set Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key (required for AI post-processing)
   - `PYTHONUNBUFFERED`: `1` (for logging)
   - `TESSDATA_PREFIX`: `/usr/share/tesseract-ocr/4/tessdata` (auto-set in Dockerfile)
   - (Optional) `FLASK_ENV`: `production`

5. **Deploy**: Click "Create Web Service"

Your app will be live at `https://your-service-name.onrender.com`

**Note**: The app uses headless OpenCV and optimized Gunicorn settings (--timeout 300s, --workers 1) for reliable OCR processing on Render's infrastructure.

### Other Deployment Options

For other platforms:

1. Set `FLASK_ENV=production` in `.env`
2. Use a production WSGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. Set up nginx as reverse proxy
4. Use PostgreSQL instead of SQLite
5. Enable HTTPS with SSL certificates

## 📝 API Response Format

```json
{
  "success": true,
  "document_id": 1,
  "extracted_fields": {
    "patta_id": {
      "value": "FRA-2023-MP-001",
      "confidence": 0.92,
      "source": "Regex",
      "raw": "FRA/2023/MP/001"
    },
    "holder_name": {
      "value": "Ram Kumar",
      "confidence": 0.85,
      "source": "Gemini",
      "raw": "राम कुमार"
    },
    "area": {
      "value": 2.5,
      "unit": "ha",
      "confidence": 0.88,
      "note": "converted from acres"
    }
  },
  "needs_verification": false
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 💡 Support

For issues or questions:
- Create an issue on GitHub
- Email: support@fra-ocr.example.com

## 🎯 Roadmap

- [ ] Add support for more Indian languages
- [ ] Implement ML-based field validation
- [ ] Add bulk export features
- [ ] Mobile app development
- [ ] Integration with government databases

---

**Built with ❤️ for digitizing Forest Rights in India**
